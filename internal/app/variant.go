package app

import (
	"context"
	"errors"
	"fmt"
	"io"
	"strings"

	"github.com/spf13/cobra"

	"poc-rag/internal/config"
	"poc-rag/internal/domain/variant"
	infraqdrant "poc-rag/internal/infrastructure/qdrant"
)

func newVariantCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "variant",
		Short: "Manage index variants",
	}
	cmd.AddCommand(newVariantDropCmd())
	return cmd
}

// newVariantDropCmd deliberately does NOT go through modules()/fx: the fx
// graph runs Setup (and with it the §9.1.1 drift check) inside the Registry
// constructor, and this command's primary use case is recovering from
// ErrConfigDrift — where that startup path refuses. It wires config, client,
// and repos directly, and takes the name as a raw string rather than
// resolving it through the config-declared registry, so a variant that is no
// longer (or was never) in the config can still be dropped.
func newVariantDropCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "drop <name>",
		Short: "Drop a variant's chunk collection and registry entry (drift recovery; no drift check)",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			cfg, err := config.Load(configPath)
			if err != nil {
				return err
			}
			c, err := infraqdrant.NewClient(infraqdrant.Options{
				Host: cfg.Qdrant.Host, GRPCPort: cfg.Qdrant.GRPCPort,
				APIKey: cfg.Qdrant.APIKey, UseTLS: cfg.Qdrant.UseTLS,
			})
			if err != nil {
				return err
			}
			defer c.Close()
			return runVariantDrop(cmd.Context(), c, args[0], cmd.OutOrStdout())
		},
	}
}

// runVariantDrop removes both artifacts of a variant: the chunks_<name>
// collection and its registry point in `variants`. Idempotent — dropping a
// name with no collection and/or no registry point (or on a Qdrant that was
// never migrated at all) succeeds quietly. The pre-checks exist only to
// report precisely what was dropped; the deletes themselves are safe either
// way (DeleteByVariant guards with CollectionExists, point deletes are
// no-ops for absent points).
func runVariantDrop(ctx context.Context, c *infraqdrant.Client, name string, out io.Writer) error {
	var dropped []string

	collection := variant.CollectionPrefix + name
	hadCollection, err := c.Qdrant().CollectionExists(ctx, collection)
	if err != nil {
		return fmt.Errorf("checking collection %s: %w", collection, err)
	}
	if err := infraqdrant.NewChunkRepository(c).DeleteByVariant(ctx, name); err != nil {
		return err
	}
	if hadCollection {
		dropped = append(dropped, "collection "+collection)
	}

	// Guard the registry side on the collection's existence: a point delete
	// against an absent `variants` collection is NotFound, not a no-op.
	registryExists, err := c.Qdrant().CollectionExists(ctx, infraqdrant.VariantsCollection)
	if err != nil {
		return fmt.Errorf("checking collection %s: %w", infraqdrant.VariantsCollection, err)
	}
	if registryExists {
		repo := infraqdrant.NewVariantRepository(c)
		switch _, err := repo.Load(ctx, name); {
		case err == nil:
			if err := repo.Delete(ctx, name); err != nil {
				return err
			}
			dropped = append(dropped, "registry entry")
		case !errors.Is(err, variant.ErrUnknown):
			return err
		}
	}

	if len(dropped) == 0 {
		fmt.Fprintf(out, "variant %q: nothing to drop\n", name)
	} else {
		fmt.Fprintf(out, "variant %q: dropped %s\n", name, strings.Join(dropped, " and "))
	}
	return nil
}
