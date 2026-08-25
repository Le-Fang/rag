package search

import (
	"net/http"

	"github.com/gin-gonic/gin"

	"poc-rag/internal/application/apperror"
)

type Handler struct {
	svc *Service
}

func NewHandler(svc *Service) *Handler { return &Handler{svc: svc} }

type searchRequestDTO struct {
	Query       string `json:"query"`
	Variant     string `json:"variant"`
	Mode        string `json:"mode,omitempty"`
	TopK        int    `json:"top_k,omitempty"`
	AnnTopK     int    `json:"ann_top_k,omitempty"`
	LexicalTopK int    `json:"lexical_top_k,omitempty"`
	Rerank      *bool  `json:"rerank,omitempty"`
}

type resultDTO struct {
	Rank          int      `json:"rank"`
	ParentChunkID string   `json:"parent_chunk_id"`
	DocumentID    string   `json:"document_id"`
	Text          string   `json:"text"`
	ChildChunkID  string   `json:"child_chunk_id"`
	ChildText     string   `json:"child_text"`
	Title         string   `json:"title,omitempty"`
	Channels      []string `json:"channels"`
	DenseScore    *float64 `json:"dense_score,omitempty"`
	DenseRank     *int     `json:"dense_rank,omitempty"`
	LexicalScore  *float64 `json:"lexical_score,omitempty"`
	LexicalRank   *int     `json:"lexical_rank,omitempty"`
	RRFScore      float64  `json:"rrf_score"`
	RerankScore   *float64 `json:"rerank_score,omitempty"`
}

type statsDTO struct {
	Mode              string           `json:"mode"`
	DenseCandidates   int              `json:"dense_candidates"`
	LexicalCandidates int              `json:"lexical_candidates"`
	FusedCandidates   int              `json:"fused_candidates"`
	Parents           int              `json:"parents"`
	Reranked          bool             `json:"reranked"`
	TimingsMs         map[string]int64 `json:"timings_ms"`
}

// Search handles POST /v1/search (§8).
func (h *Handler) Search(c *gin.Context) {
	var dto searchRequestDTO
	if err := c.ShouldBindJSON(&dto); err != nil {
		apperror.Respond(c, apperror.FromBindError(err))
		return
	}
	resp, err := h.svc.Search(c.Request.Context(), Request{
		Query: dto.Query, Variant: dto.Variant, Mode: dto.Mode,
		TopK: dto.TopK, AnnTopK: dto.AnnTopK, LexicalTopK: dto.LexicalTopK,
		Rerank: dto.Rerank,
	})
	if err != nil {
		apperror.Respond(c, err)
		return
	}
	results := make([]resultDTO, len(resp.Results))
	for i, r := range resp.Results {
		results[i] = resultDTO{
			Rank: r.Rank, ParentChunkID: r.ParentChunkID, DocumentID: r.DocumentID,
			Text: r.Text, ChildChunkID: r.ChildChunkID, ChildText: r.ChildText,
			Title:    r.Title,
			Channels: r.Channels, DenseScore: r.DenseScore, DenseRank: r.DenseRank,
			LexicalScore: r.LexicalScore, LexicalRank: r.LexicalRank,
			RRFScore: r.RRFScore, RerankScore: r.RerankScore,
		}
	}
	c.JSON(http.StatusOK, gin.H{
		"results": results,
		"stats": statsDTO{
			Mode: resp.Stats.Mode, DenseCandidates: resp.Stats.DenseCandidates,
			LexicalCandidates: resp.Stats.LexicalCandidates, FusedCandidates: resp.Stats.FusedCandidates,
			Parents: resp.Stats.Parents, Reranked: resp.Stats.Reranked, TimingsMs: resp.Stats.TimingsMs,
		},
	})
}
