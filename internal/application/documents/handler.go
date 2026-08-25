package documents

import (
	"net/http"
	"time"

	"github.com/gin-gonic/gin"

	"poc-rag/internal/application/apperror"
	"poc-rag/internal/domain/document"
)

type Handler struct {
	svc *Service
}

func NewHandler(svc *Service) *Handler { return &Handler{svc: svc} }

type documentDTO struct {
	ID        string     `json:"id,omitempty"`
	Title     string     `json:"title"`
	Content   string     `json:"content"`
	CreatedAt *time.Time `json:"created_at,omitempty"`
	UpdatedAt *time.Time `json:"updated_at,omitempty"`
}

func (d documentDTO) toDomain() *document.Document {
	return &document.Document{ID: d.ID, Title: d.Title, Content: d.Content}
}

func fromDomain(doc *document.Document) documentDTO {
	dto := documentDTO{ID: doc.ID, Title: doc.Title, Content: doc.Content}
	// omitempty only omits a nil pointer — a zero time.Time would otherwise
	// still serialize as "0001-01-01T00:00:00Z".
	if !doc.CreatedAt.IsZero() {
		dto.CreatedAt = &doc.CreatedAt
	}
	if !doc.UpdatedAt.IsZero() {
		dto.UpdatedAt = &doc.UpdatedAt
	}
	return dto
}

// Save handles POST /v1/documents[?variant=X] (§8).
func (h *Handler) Save(c *gin.Context) {
	var dto documentDTO
	if err := c.ShouldBindJSON(&dto); err != nil {
		apperror.Respond(c, apperror.FromBindError(err))
		return
	}
	saved, created, err := h.svc.Save(c.Request.Context(), dto.toDomain(), c.Query("variant"))
	if err != nil {
		apperror.Respond(c, err)
		return
	}
	status := http.StatusOK // replace of an existing id
	if created {
		status = http.StatusCreated
	}
	c.JSON(status, fromDomain(saved))
}

// Get handles GET /v1/documents/:id.
func (h *Handler) Get(c *gin.Context) {
	doc, err := h.svc.Get(c.Request.Context(), c.Param("id"))
	if err != nil {
		apperror.Respond(c, err)
		return
	}
	c.JSON(http.StatusOK, fromDomain(doc))
}

// Delete handles DELETE /v1/documents/:id.
func (h *Handler) Delete(c *gin.Context) {
	if err := h.svc.Delete(c.Request.Context(), c.Param("id")); err != nil {
		apperror.Respond(c, err)
		return
	}
	c.Status(http.StatusNoContent)
}
