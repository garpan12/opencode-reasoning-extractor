package pipeline

import (
	"fmt"
	"io"

	"opencode-reasoning-extractor/internal/classify"
	"opencode-reasoning-extractor/internal/compliance"
	"opencode-reasoning-extractor/internal/convert"
	"opencode-reasoning-extractor/internal/format"
	"opencode-reasoning-extractor/internal/ingest"
	"opencode-reasoning-extractor/internal/model"
	"opencode-reasoning-extractor/internal/redact"
)

// runJSONL extracts from an external SessionRecord dataset (our converter
// output) instead of an opencode store. It reuses every downstream stage:
// classification, redaction, refusal filtering, per-turn records and RL meta.
func runJSONL(opts Options) (*Stats, error) {
	files, err := ingest.Files(opts.Input)
	if err != nil {
		return nil, err
	}
	if len(files) == 0 {
		return nil, fmt.Errorf("no .jsonl session files found under %s", opts.Input)
	}

	rawCatalog, err := classify.RawCatalog(opts.Catalog)
	if err != nil {
		return nil, err
	}
	cat, err := classify.LoadCatalog(opts.Catalog)
	if err != nil {
		return nil, err
	}
	classifier, err := classify.New(cat)
	if err != nil {
		return nil, err
	}
	detector, err := compliance.Default()
	if err != nil {
		return nil, err
	}
	rawRefusals := compliance.RawCatalog()

	stats := &Stats{Topics: map[string]int{}, Models: map[string]int{}, Agents: map[string]int{}}

	if err := prepareOutput(opts); err != nil {
		return nil, err
	}
	var out *format.Output
	if !opts.DryRun {
		out, err = format.NewOutput(opts.Output, opts.Strict)
		if err != nil {
			return nil, err
		}
		defer out.Close()
	}

	exported, err := loadExported(opts)
	if err != nil {
		return nil, err
	}

	modelFilter := splitFilter(opts.Models)
	agentFilter := splitFilter(opts.Agents)

	reader := ingest.NewReader(files)
	defer reader.Close()

	processed := 0
	for {
		rec, err := reader.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}

		sess, conv := ingest.ToConversation(rec)
		if !matches(sess, modelFilter, agentFilter, opts.Session) {
			continue
		}
		if exported[sess.ID] {
			continue
		}
		if opts.Limit > 0 && processed >= opts.Limit {
			break
		}
		if len(conv.Messages) == 0 {
			continue
		}

		if !opts.NoRedact {
			r := redact.New()
			for i := range conv.Messages {
				conv.Messages[i].Content = r.Redact(conv.Messages[i].Content)
				conv.Messages[i].ReasoningContent = r.Redact(conv.Messages[i].ReasoningContent)
			}
			conv.Meta.Redactions = r.Total()
		}

		doc := buildDoc(sess, conv)
		topics, scores := classifier.Classify(doc)

		if verdict := detector.Evaluate(conv); verdict.Refused {
			stats.Refusals++
			if opts.Verbose {
				fmt.Printf("  %s  skipped: refusal detected (score=%.1f matches=%v)\n", sess.ID, verdict.Score, verdict.Matches)
			}
			continue
		}
		if opts.MaxTopics > 0 && len(topics) > opts.MaxTopics {
			topics = topics[:opts.MaxTopics]
		}
		processed++

		stats.Sessions++
		for _, t := range topics {
			stats.Topics[t]++
		}
		stats.Models[sess.Model.ID]++
		stats.Agents[sess.Agent]++
		stats.ToolCalls += conv.Meta.ToolCalls
		stats.ToolErrors += conv.Meta.ToolErrors
		stats.Redactions += conv.Meta.Redactions

		if opts.DryRun {
			exported[sess.ID] = true
			if opts.Verbose {
				fmt.Printf("  %s  agent=%-8s model=%-28s topics=%v\n", sess.ID, sess.Agent, sess.Model.ID, topics)
			}
			continue
		}

		meta := jsonlRecordMeta(rec, scores)
		tree := "sft"
		if !opts.NoTurns {
			for _, turn := range convert.TurnRecords(conv, topics, scores, meta, opts.TurnContext) {
				for _, topic := range topics {
					if err := out.WriteTurn(tree, topic, turn); err != nil {
						return nil, err
					}
				}
			}
		}
		if !opts.NoSessions {
			for _, topic := range topics {
				if err := out.WriteSession(tree, topic, model.SessionRecord{
					ID: sess.ID, Topic: topics, Messages: conv.Messages, Meta: meta,
				}); err != nil {
					return nil, err
				}
			}
		}
		if err := out.WriteRL(buildRL(conv, sess, topics, scores)); err != nil {
			return nil, err
		}
		exported[sess.ID] = true

		if opts.Verbose {
			fmt.Printf("  %s -> %s/%s (%d msgs)\n", sess.ID, tree, topics, len(conv.Messages))
		}
	}

	if !opts.DryRun && out != nil {
		if err := out.WriteCatalog(rawCatalog); err != nil {
			return nil, err
		}
		if err := out.WriteRefusalCatalog(rawRefusals); err != nil {
			return nil, err
		}
		if err := out.WriteManifest(buildManifest(opts, rawCatalog, rawRefusals, exported, stats)); err != nil {
			return nil, err
		}
	}
	return stats, nil
}

// jsonlRecordMeta keeps the dataset's own metadata and attaches the classifier
// scores, matching the shape recordMeta() produces for the store path.
func jsonlRecordMeta(rec *model.SessionRecord, scores map[string]float64) model.RecordMeta {
	meta := rec.Meta
	meta.TopicScores = scores
	return meta
}
