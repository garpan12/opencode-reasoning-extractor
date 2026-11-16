// Package ingest reads externally-prepared session datasets (our own
// SessionRecord JSONL, e.g. from qwen-vast/data/hf_to_extractor.py) and turns
// them into the same model.Conversation the opencode store path produces, so
// the rest of the pipeline (classify, redact, refusal filter, turns, RL) can be
// reused unchanged.
package ingest

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"

	"opencode-reasoning-extractor/internal/model"
)

// LooksLikeJSONL reports whether input is an external SessionRecord dataset
// rather than an opencode data directory. It is either a .jsonl file or a
// directory laid out like our converter output (manifest.json + sft/).
func LooksLikeJSONL(input string) bool {
	fi, err := os.Stat(input)
	if err != nil {
		return false
	}
	if !fi.IsDir() {
		return filepath.Ext(input) == ".jsonl"
	}
	if _, err := os.Stat(filepath.Join(input, "manifest.json")); err == nil {
		if _, err := os.Stat(filepath.Join(input, "sft")); err == nil {
			return true
		}
	}
	// A directory that directly holds sessions.jsonl files also qualifies.
	matches, _ := filepath.Glob(filepath.Join(input, "*.jsonl"))
	return len(matches) > 0
}

// Files returns the sessions.jsonl files to read for the given input. A single
// .jsonl file returns itself; a directory returns every sessions.jsonl beneath
// it (sorted), falling back to any *.jsonl directly inside it.
func Files(input string) ([]string, error) {
	fi, err := os.Stat(input)
	if err != nil {
		return nil, err
	}
	if !fi.IsDir() {
		return []string{input}, nil
	}
	var out []string
	err = filepath.WalkDir(input, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !d.IsDir() && d.Name() == "sessions.jsonl" {
			out = append(out, path)
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	if len(out) == 0 {
		out, err = filepath.Glob(filepath.Join(input, "*.jsonl"))
		if err != nil {
			return nil, err
		}
	}
	sort.Strings(out)
	return out, nil
}

// Reader streams SessionRecords from one or more JSONL files.
type Reader struct {
	files []string
	fi    int
	f     *os.File
	dec   *json.Decoder
}

// NewReader opens a reader over the given files.
func NewReader(files []string) *Reader {
	return &Reader{files: files}
}

// Next returns the next record, or io.EOF when exhausted. Malformed lines are
// skipped. The returned pointer is only valid until the following call.
func (r *Reader) Next() (*model.SessionRecord, error) {
	for {
		if r.dec == nil {
			if r.fi >= len(r.files) {
				return nil, io.EOF
			}
			f, err := os.Open(r.files[r.fi])
			r.fi++
			if err != nil {
				return nil, err
			}
			r.f = f
			r.dec = json.NewDecoder(f)
		}
		var rec model.SessionRecord
		err := r.dec.Decode(&rec)
		if err != nil {
			// EOF or a malformed value: the decoder can't reliably resync mid
			// stream, so finish this file and move to the next one.
			r.f.Close()
			r.f, r.dec = nil, nil
			continue
		}
		return &rec, nil
	}
}

// Close releases any open file.
func (r *Reader) Close() error {
	if r.f != nil {
		return r.f.Close()
	}
	return nil
}

// ToConversation reconstructs a synthetic Session and a Conversation from a
// record. Metadata that the opencode store would carry (tokens, cost, tool
// errors) is taken from the record's meta when present, else left zero.
func ToConversation(rec *model.SessionRecord) (model.Session, model.Conversation) {
	sess := model.Session{
		ID:          rec.ID,
		Agent:       rec.Meta.Agent,
		Model:       model.ModelRef{ID: rec.Meta.Model, ProviderID: rec.Meta.Provider},
		Directory:   rec.Meta.Directory,
		ParentID:    rec.Meta.ParentID,
		ProjectID:   rec.Meta.Project,
		TimeCreated: rec.Meta.TimeCreated,
		TimeUpdated: rec.Meta.TimeUpdated,
	}
	conv := model.Conversation{SessionID: rec.ID}
	conv.Meta = model.ConversationMeta{
		Agent:         sess.Agent,
		ModelID:       sess.Model.ID,
		ProviderID:    sess.Model.ProviderID,
		ParentID:      sess.ParentID,
		ProjectID:     sess.ProjectID,
		Directory:     sess.Directory,
		TimeCreated:   sess.TimeCreated,
		TimeUpdated:   sess.TimeUpdated,
		FinishReasons: map[string]int{},
	}
	seen := map[string]bool{}
	for _, m := range rec.Messages {
		conv.Messages = append(conv.Messages, m)
		switch m.Role {
		case "system":
			conv.Meta.SystemPrompt = true
		case "user":
			conv.Meta.UserTurns++
		case "assistant":
			idx := len(conv.Messages) - 1
			conv.Messages[idx].SourceID = fmt.Sprintf("assistant-%d", len(conv.AssistantTurns))
			conv.AssistantTurns = append(conv.AssistantTurns, idx)
			if m.ReasoningContent != "" {
				conv.Meta.ReasoningTurns++
			}
			for _, tc := range m.ToolCalls {
				conv.Meta.ToolCalls++
				if !seen[tc.Function.Name] {
					seen[tc.Function.Name] = true
					conv.Meta.ToolNames = append(conv.Meta.ToolNames, tc.Function.Name)
				}
			}
		}
	}
	if rec.Meta.Tokens != nil {
		sess.Tokens = *rec.Meta.Tokens
		conv.Meta.Tokens = *rec.Meta.Tokens
	}
	if rec.Meta.Cost != nil {
		sess.Cost = *rec.Meta.Cost
		conv.Meta.Cost = *rec.Meta.Cost
	}
	return sess, conv
}
