package app

import (
	"github.com/spf13/cobra"
	"go.uber.org/fx"

	"poc-rag/internal/config"
	"poc-rag/internal/infrastructure/httpserver"
)

func newServerCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "server",
		Short: "Run the HTTP API",
		RunE: func(cmd *cobra.Command, args []string) error {
			app := fx.New(
				modules(configPath),
				fx.Provide(httpserver.NewEngine),
				fx.Provide(func(cfg *config.Config) httpserver.Options {
					return httpserver.Options{
						Port:         cfg.Server.HTTPPort,
						ReadTimeout:  cfg.Server.ReadTimeout,
						WriteTimeout: cfg.Server.WriteTimeout,
					}
				}),
				fx.Invoke(httpserver.Register),
			)
			if err := app.Err(); err != nil {
				return err
			}
			app.Run() // blocks until SIGINT/SIGTERM
			return nil
		},
	}
}
