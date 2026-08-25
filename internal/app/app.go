package app

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"github.com/spf13/cobra"
)

var configPath string

func Execute() {
	// Ctrl+C / SIGTERM cancels cmd.Context() so long-running commands (ingest)
	// abort promptly. The server command is unaffected: fx's app.Run() has its
	// own signal handling, and os/signal delivers to every registered channel.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	root := &cobra.Command{
		Use:   "poc-rag",
		Short: "RAG retrieval-quality POC backed by Qdrant",
		// A runtime failure (bad config, Qdrant down, drift) is not a usage
		// mistake: cobra would otherwise dump the whole help text after the
		// error and print the error a second time itself. We own both.
		SilenceUsage:  true,
		SilenceErrors: true,
	}
	root.PersistentFlags().StringVar(&configPath, "config", ".apprc.yaml", "path to config file")

	root.AddCommand(newServerCmd(), newMigrateCmd(), newIngestCmd(), newVariantCmd())

	if err := root.ExecuteContext(ctx); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
