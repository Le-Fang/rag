// Package httpserver wires gin routes to handlers and manages the listener
// through the Fx lifecycle.
package httpserver

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/gin-gonic/gin/binding"
	"go.uber.org/fx"

	"poc-rag/internal/application/documents"
	"poc-rag/internal/application/search"
)

// maxRequestBodyBytes caps request bodies at 64MiB, matching the raised
// gRPC recv cap — document content bodies are legitimately large.
const maxRequestBodyBytes = 64 << 20

// ginGlobalsOnce guards gin's package-global mutations below so multiple
// engines built in the same test binary (e.g. Task 19's e2e suite) don't
// race or redundantly reassign them.
var ginGlobalsOnce sync.Once

type Options struct {
	Port         int
	ReadTimeout  time.Duration
	WriteTimeout time.Duration
}

func NewEngine(docs *documents.Handler, srch *search.Handler, log *slog.Logger) *gin.Engine {
	return newEngine(docs, srch, log, maxRequestBodyBytes)
}

// newEngine takes the body cap as a parameter so tests can exercise the
// 413 path without pushing 64MiB through httptest; production always enters
// through NewEngine with maxRequestBodyBytes.
func newEngine(docs *documents.Handler, srch *search.Handler, log *slog.Logger, bodyLimitBytes int64) *gin.Engine {
	ginGlobalsOnce.Do(func() {
		// Unknown JSON request fields 400 instead of being silently ignored —
		// a typo'd field must not silently yield a wrong-config comparison in
		// this eval-focused POC.
		binding.EnableDecoderDisallowUnknownFields = true
		gin.SetMode(gin.ReleaseMode)
	})

	e := gin.New()
	e.Use(gin.Recovery())
	e.Use(func(c *gin.Context) {
		start := time.Now()
		c.Next()
		log.InfoContext(c.Request.Context(), "request",
			"method", c.Request.Method, "path", c.FullPath(),
			"status", c.Writer.Status(), "latency_ms", time.Since(start).Milliseconds())
	})
	e.Use(bodyLimit(bodyLimitBytes))

	// Liveness only: no Qdrant ping — reachability fails startup via Fx (§8).
	e.GET("/healthz", func(c *gin.Context) { c.JSON(http.StatusOK, gin.H{"status": "ok"}) })

	v1 := e.Group("/v1")
	v1.POST("/documents", docs.Save)
	v1.GET("/documents/:id", docs.Get)
	v1.DELETE("/documents/:id", docs.Delete)
	v1.POST("/search", srch.Search)
	return e
}

// bodyLimit caps request bodies. When the cap trips, the handlers' JSON binds
// surface a *http.MaxBytesError, which apperror.FromBindError maps to a 413
// naming the limit.
func bodyLimit(n int64) gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, n)
		c.Next()
	}
}

func Register(lc fx.Lifecycle, engine *gin.Engine, opts Options, log *slog.Logger) {
	srv := &http.Server{
		Addr:         fmt.Sprintf(":%d", opts.Port),
		Handler:      engine,
		ReadTimeout:  opts.ReadTimeout,
		WriteTimeout: opts.WriteTimeout,
	}
	lc.Append(fx.Hook{
		OnStart: func(ctx context.Context) error {
			// Bind synchronously so a port conflict fails startup instead of
			// only surfacing later via an async log line.
			ln, err := net.Listen("tcp", srv.Addr)
			if err != nil {
				return fmt.Errorf("listen %s: %w", srv.Addr, err)
			}
			log.InfoContext(ctx, "http server listening", "addr", srv.Addr)
			go func() {
				if err := srv.Serve(ln); err != nil && !errors.Is(err, http.ErrServerClosed) {
					log.ErrorContext(context.Background(), "http server failed", "error", err)
				}
			}()
			return nil
		},
		OnStop: func(ctx context.Context) error {
			return srv.Shutdown(ctx)
		},
	})
}
