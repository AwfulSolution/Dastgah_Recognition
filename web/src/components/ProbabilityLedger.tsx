import type { AnalysisResult, LedgerRow } from "../lib/types";

interface Group {
  family: string;
  total: number;
  rows: LedgerRow[];
}

/**
 * Probabilities grouped by mode family. Families are what the method separates
 * reliably, so the group totals carry more weight than the rows inside them.
 */
function groupByFamily(ledger: LedgerRow[]): Group[] {
  const groups = new Map<string, Group>();
  for (const row of ledger) {
    const key = row.family || row.key;
    const existing = groups.get(key);
    if (existing) {
      existing.total += row.probability;
      existing.rows.push(row);
    } else {
      groups.set(key, { family: key, total: row.probability, rows: [row] });
    }
  }
  const list = [...groups.values()];
  for (const group of list) group.rows.sort((a, b) => b.probability - a.probability);
  list.sort((a, b) => b.total - a.total);
  return list;
}

export default function ProbabilityLedger({ result }: { result: AnalysisResult }) {
  const names = new Map<string, string>();
  for (const row of result.ledger) names.set(row.key, row.short_name);
  const groups = groupByFamily(result.ledger);

  return (
    <section className="panel">
      <header className="panel-header justify-between">
        <span className="label-mono text-secondary">Modal Probability Ledger</span>
        <span className="label-mono text-outline">
          {groups.length} famil{groups.length === 1 ? "y" : "ies"} · 13 modes
        </span>
      </header>

      <div className="p-4">
        <p className="mb-3 text-[13px] leading-relaxed text-on-surface/70">
          Grouped by mode family. The family totals are the reliable reading;
          within a family the modes share a pitch collection and the split between
          them is far less certain.
        </p>

        <ol className="space-y-3">
          {groups.map((group, index) => {
            const leading = index === 0;
            const familyLabel = names.get(group.family) ?? group.family;
            const solo = group.rows.length === 1;
            return (
              <li key={group.family}>
                <div className="flex items-baseline justify-between gap-2">
                  <span className="truncate font-serif text-[14px]">
                    <span
                      className={leading ? "text-primary" : "text-on-surface/85"}
                    >
                      {solo ? familyLabel : `${familyLabel} group`}
                    </span>
                    {!solo && (
                      <span className="ml-1.5 font-mono text-[9px] uppercase tracking-wider text-outline">
                        {group.rows.length} modes
                      </span>
                    )}
                  </span>
                  <span
                    className={`shrink-0 font-mono text-[13px] ${leading ? "text-primary" : "text-on-surface-variant"}`}
                  >
                    {(group.total * 100).toFixed(1)}%
                  </span>
                </div>

                <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-surface-container-highest/60">
                  <div
                    className={`h-full rounded-full ${leading ? "bg-primary shadow-glow-gold" : "bg-secondary-container/70"}`}
                    style={{ width: `${Math.max(1.5, group.total * 100)}%` }}
                  />
                </div>

                {!solo && (
                  <ul className="mt-1.5 space-y-0.5 border-l border-hairline pl-3">
                    {group.rows.map((row) => (
                      <li
                        key={row.key}
                        className="flex items-baseline justify-between gap-2"
                      >
                        <span className="truncate font-serif text-[12px] text-on-surface/70">
                          {row.name}
                          {row.kind === "avaz" && (
                            <span className="ml-1.5 font-mono text-[9px] uppercase tracking-wider text-outline">
                              āvāz
                            </span>
                          )}
                        </span>
                        <span className="shrink-0 font-mono text-[11px] text-outline">
                          {(row.probability * 100).toFixed(1)}%
                        </span>
                      </li>
                    ))}
                  </ul>
                )}
              </li>
            );
          })}
        </ol>
      </div>
    </section>
  );
}
