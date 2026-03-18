// Kaggle Proxy - Lightweight proxy for Kaggle Models API
// This binary runs inside Kaggle Notebooks and proxies requests to Kaggle's internal Models API.
package main

import (
	"context"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/kaggle"
	_ "github.com/router-for-me/CLIProxyAPI/v6/internal/translator"
	log "github.com/sirupsen/logrus"
)

func main() {
	// Setup logging
	log.SetFormatter(&log.TextFormatter{
		FullTimestamp: true,
	})
	log.SetLevel(log.InfoLevel)

	// Load configuration
	cfg := kaggle.LoadConfig()
	log.Infof("Kaggle proxy starting with URL: %s", cfg.ProxyURL)

	// Setup Gin
	gin.SetMode(gin.ReleaseMode)
	engine := gin.New()
	engine.Use(gin.Recovery())
	engine.Use(corsMiddleware())

	// Register routes
	handler := kaggle.NewHandler(cfg)
	handler.RegisterRoutes(engine)

	// Root endpoint
	engine.GET("/", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"message": "Kaggle Proxy for CLIProxyAPIPlus",
			"endpoints": []string{
				"POST /v1/chat/completions (OpenAI)",
				"POST /v1/messages (Claude)",
				"POST /v1beta/models/* (Gemini)",
			},
		})
	})

	// Get port from environment or use default
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	// Create HTTP server
	srv := &http.Server{
		Addr:    ":" + port,
		Handler: engine,
	}

	// Start server in goroutine
	go func() {
		log.Infof("Kaggle proxy listening on :%s", port)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Failed to start server: %v", err)
		}
	}()

	// Wait for interrupt signal
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	log.Info("Shutting down server...")

	// Graceful shutdown
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		log.Fatalf("Server forced to shutdown: %v", err)
	}

	log.Info("Server stopped")
}

// corsMiddleware adds CORS headers.
func corsMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Header("Access-Control-Allow-Origin", "*")
		c.Header("Access-Control-Allow-Methods", "GET, POST, PUT, PATCH, DELETE, OPTIONS")
		c.Header("Access-Control-Allow-Headers", "*")

		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(http.StatusNoContent)
			return
		}

		c.Next()
	}
}
