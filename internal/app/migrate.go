package app

import (
	"github.com/spf13/cobra"
	"go.uber.org/fx"

	"poc-rag/internal/domain/variant"
)

func newMigrateCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "migrate",
		Short: "Ensure collections, payload indexes, and the variant registry; check drift",
		RunE: func(cmd *cobra.Command, args []string) error {
			// Setup runs inside the Registry constructor; requesting the
			// registry forces it (§5.2).
			app := fx.New(modules(configPath), fx.Invoke(func(*variant.Registry) {}))
			if err := app.Err(); err != nil {
				return err
			}
			// run the lifecycle so OnStop hooks fire (closes the Qdrant
			// client) — fx executes stop hooks only after a successful Start
			if err := app.Start(cmd.Context()); err != nil {
				return err
			}
			return app.Stop(cmd.Context())
		},
	}
}
