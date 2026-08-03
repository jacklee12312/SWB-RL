"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  useState,
} from "react";

const API_BASE =
  process.env.NEXT_PUBLIC_SWB_API_BASE ?? "http://127.0.0.1:8765";

type MatchAction = {
  id: number;
  kind: string;
  label: string;
  source_entity_id?: number | null;
  target_entity_id?: number | null;
};

type UnionBurstProgress = {
  kind: "union_burst" | "super_skybound_art";
  label: "奥义" | "解放奥义";
  gauge: number;
  threshold: number;
  remaining: number;
  ready: boolean;
};

type CardView = {
  index?: number;
  entity_id: number;
  card_id: number;
  name: string;
  type: string;
  cost: number;
  printed_cost?: number;
  attack?: number | null;
  health?: number | null;
  max_health?: number | null;
  countdown?: number | null;
  earth_sigils?: number;
  evolved?: boolean;
  super_evolved?: boolean;
  can_attack?: boolean;
  keywords: string[];
  union_bursts?: UnionBurstProgress[];
  image_url?: string | null;
};

type FaithView = {
  entity_id: number;
  faith_id: string;
  source_card_id: number;
  source_name: string;
  image_url?: string | null;
  value: number;
};

type EmblemView = {
  entity_id: number;
  emblem_id: string;
  source_card_id: number;
  source_name: string;
  image_url?: string | null;
  countdown: number | null;
};

type PlayerView = {
  player_index: number;
  role: "human" | "ai";
  class_id: number;
  class_name: string;
  health: number;
  max_health: number;
  mana: number;
  max_mana: number;
  extra_pp_available: boolean;
  evolution_points: number;
  super_evolution_points: number;
  shadows: number;
  cooperation: number;
  cards_played_this_turn: number;
  overflow_active: boolean;
  earth_sigils: number;
  leader_area_used: number;
  leader_area_limit: number;
  deck_count: number;
  hand_count: number;
  graveyard_count: number;
  banished_count: number;
  board: CardView[];
  hand: CardView[] | null;
  faiths: FaithView[];
  emblems: EmblemView[];
};

type AnimationCue = {
  id: string;
  action_sequence: number;
  kind: string;
  title: string;
  detail: string | null;
  actor_player: number | null;
  source_entity_id: number | null;
  source_name: string | null;
  target_entity_id: number | null;
  target_name: string | null;
  amount: number;
  duration_ms: number;
};

type DeckOption = {
  name: string;
  display_name: string;
  class_id: number;
  sha256: string;
};

type ModelOption = {
  id: string;
  display_name: string;
  group: string;
  filename: string;
  size_bytes: number;
};

type MatchState = {
  seed: number;
  match_id: string;
  deck: { name: string; display_name: string; sha256: string };
  human_deck: DeckOption;
  ai_deck: DeckOption;
  specialist_deck: DeckOption | null;
  available_decks: DeckOption[];
  model?: ModelOption;
  available_models?: ModelOption[];
  checkpoint: string;
  warnings: string[];
  human_player: number;
  ai_player: number;
  current_player: number;
  decision_player: number;
  first_player: number;
  turn: number;
  phase: string;
  terminated: boolean;
  truncated: boolean;
  winner: number | null;
  human_turn: boolean;
  players: PlayerView[];
  actions: MatchAction[];
  pending_choice: {
    prompt: string;
    kind: string;
    target_count: number;
    selected_count: number;
  } | null;
  last_ai_actions: string[];
  animation_batch_id: string;
  animation_batch: AnimationCue[];
  logs: string[];
};

type HistorySummary = {
  match_id: string;
  created_at?: string;
  updated_at?: string;
  seed?: number;
  status: string;
  winner?: number | null;
  human_player?: number;
  turn?: number;
  phase?: string;
  action_count?: number;
  deck_display_name?: string;
  checkpoint?: string;
  error?: string;
};

type HistoryAction = {
  sequence: number;
  actor_role: "human" | "ai";
  player_index: number;
  action_id: number;
  action: MatchAction;
  decision?: {
    type: "human" | "ppo_argmax";
    policy_architecture: string | null;
    selected_action_id: number;
    selected_probability: number | null;
    value: number | null;
    legal_actions: Array<
      MatchAction & {
        selected: boolean;
        logit: number | null;
        probability: number | null;
      }
    >;
  };
  before?: {
    phase: string;
    players: PlayerView[];
  };
  logs: string[];
  animations: AnimationCue[];
};

type HistoryRecord = {
  match_id: string;
  created_at: string;
  updated_at: string;
  seed: number;
  human_player: number;
  status: string;
  winner: number | null;
  turn: number;
  phase: string;
  checkpoint: string;
  deck: { display_name: string };
  actions: HistoryAction[];
  logs: string[];
};

function historyRevealsPrivateInformation(record: HistoryRecord) {
  return record.status !== "ongoing";
}

function historyActionLabel(record: HistoryRecord, action: HistoryAction) {
  if (
    historyRevealsPrivateInformation(record) ||
    action.player_index === record.human_player ||
    action.before?.phase !== "mulligan" ||
    action.action.kind !== "choice"
  ) {
    return action.action.label;
  }
  return "完成起手换牌";
}

function historyCue(record: HistoryRecord, cue: AnimationCue): AnimationCue {
  if (
    historyRevealsPrivateInformation(record) ||
    cue.actor_player == null ||
    cue.actor_player === record.human_player ||
    cue.kind !== "draw"
  ) {
    return cue;
  }
  return {
    ...cue,
    title: `玩家 ${cue.actor_player + 1} 抽取一张卡`,
    detail: null,
    source_entity_id: null,
    source_name: null,
  };
}

function historyLog(record: HistoryRecord, original: string) {
  if (historyRevealsPrivateInformation(record)) return original;
  const hiddenPlayer = 1 - record.human_player;
  const marker = `[玩家 ${hiddenPlayer + 1}]`;
  if (!original.includes(marker)) return original;
  if (original.includes("起手：")) {
    return `${original.split("起手：", 1)[0]}起手：隐藏卡牌`;
  }
  if (original.includes("回合抽牌：")) {
    return `${original.split("回合抽牌：", 1)[0]}回合抽牌：1 张`;
  }
  return original;
}

function formatPolicyProbability(value: number | null, digits = 1) {
  if (value == null) return "—";
  return `${(value * 100).toFixed(digits)}%`;
}

class RequestError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new RequestError(payload.error || `请求失败：${response.status}`, response.status);
  }
  return payload;
}

function imageUrl(path?: string | null) {
  return path ? `${API_BASE}${path}` : "";
}

function formatHistoryTime(value?: string) {
  if (!value) return "时间未知";
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(new Date(value));
}

function historyStatus(summary: Pick<HistorySummary, "status" | "winner" | "human_player">) {
  if (summary.status === "ongoing") return "进行中";
  if (summary.status === "abandoned") return "未完成";
  if (summary.status === "truncated") return "达到上限";
  if (summary.status === "corrupt") return "记录损坏";
  if (summary.winner == null) return "平局";
  return summary.winner === summary.human_player ? "你获胜" : "AI 获胜";
}

function CardTile({
  card,
  selected,
  actionable,
  onClick,
  compact = false,
}: {
  card: CardView;
  selected: boolean;
  actionable: boolean;
  onClick: () => void;
  compact?: boolean;
}) {
  const stats =
    card.type === "随从" && card.attack != null
      ? `${card.attack} / ${card.health}`
      : card.countdown != null
        ? `倒数 ${card.countdown}`
        : card.type;
  return (
    <button
      className={`card-tile ${compact ? "compact" : ""} ${
        actionable ? "actionable" : ""
      } ${selected ? "selected" : ""}`}
      onClick={onClick}
      type="button"
      aria-label={`${card.name}，费用 ${card.cost}`}
    >
      <span className="card-art-wrap">
        {card.image_url ? (
          /* Card art comes from the local Python process. */
          // eslint-disable-next-line @next/next/no-img-element
          <img className="card-art" src={imageUrl(card.image_url)} alt="" />
        ) : (
          <span className="card-art-missing">NO ART</span>
        )}
        <span className="cost-orb">{card.cost}</span>
        {card.super_evolved ? (
          <span className="evolve-tag super">超进化</span>
        ) : card.evolved ? (
          <span className="evolve-tag">进化</span>
        ) : null}
        <span className="card-stats">{stats}</span>
      </span>
      <span className="card-name">{card.name}</span>
      {card.union_bursts && card.union_bursts.length > 0 && (
        <span className="card-burst-progress" aria-label="奥义进度">
          {card.union_bursts.map((burst) => (
            <span
              className={`card-burst-badge ${burst.ready ? "ready" : ""}`}
              key={burst.kind}
              title={
                burst.ready
                  ? `${burst.label}已就绪`
                  : `${burst.label}还差 ${burst.remaining}`
              }
            >
              {burst.label} {Math.min(burst.gauge, burst.threshold)}/{burst.threshold}
            </span>
          ))}
        </span>
      )}
      {card.keywords.length > 0 && (
        <span className="card-keywords">{card.keywords.slice(0, 3).join(" · ")}</span>
      )}
    </button>
  );
}

function LeaderPanel({
  player,
  active,
  first,
  label,
}: {
  player: PlayerView;
  active: boolean;
  first: boolean;
  label: string;
}) {
  const mechanics = [
    { classId: 1, label: "连击", value: player.cards_played_this_turn },
    { classId: 2, label: "协作", value: player.cooperation },
    {
      classId: 4,
      label: "觉醒",
      value: player.overflow_active ? "已觉醒" : `${player.max_mana}/7`,
      active: player.overflow_active,
    },
    { classId: 5, label: "墓影", value: player.shadows },
    { classId: 3, label: "土之印", value: player.earth_sigils },
    {
      classId: 6,
      label: "信仰/纹章",
      value: `${player.leader_area_used}/${player.leader_area_limit}`,
    },
  ];
  const leaderContents = [
    ...player.faiths.map((faith) => ({
      entity_id: faith.entity_id,
      kind: "faith",
      name: faith.source_name,
      image_url: faith.image_url,
      value: `信仰 ${faith.value}`,
    })),
    ...player.emblems.map((emblem) => ({
      entity_id: emblem.entity_id,
      kind: "emblem",
      name: emblem.source_name,
      image_url: emblem.image_url,
      value: emblem.countdown == null ? "纹章" : `纹章 ${emblem.countdown}`,
    })),
  ].sort((left, right) => left.entity_id - right.entity_id);

  return (
    <section className={`leader-panel ${active ? "active" : ""}`}>
      <div className="leader-heading">
        <span className="role-kicker">{label}</span>
        <strong>{player.class_name}</strong>
        <span className="seat-badge">{first ? "先手" : "后手"}</span>
      </div>
      <div className="leader-core">
        <span className="health-gem">
          <b>{player.health}</b>
          <small>/{player.max_health}</small>
        </span>
        <span className="pp-readout">
          PP <b>{player.mana}</b> / {player.max_mana}
          {player.extra_pp_available && <i>+1</i>}
        </span>
        <span>EP {player.evolution_points}</span>
        <span>SEP {player.super_evolution_points}</span>
      </div>

      <div className="zone-counters" aria-label="卡牌区域">
        <span><i>牌库</i><b>{player.deck_count}</b></span>
        <span><i>手牌</i><b>{player.hand_count}</b></span>
        <span><i>墓场</i><b>{player.graveyard_count}</b></span>
        {player.banished_count > 0 && <span><i>消失</i><b>{player.banished_count}</b></span>}
      </div>

      <div className="class-mechanics" aria-label="职业特性">
        {mechanics.map((mechanic) => (
          <span
            key={mechanic.label}
            className={`${mechanic.classId === player.class_id ? "primary" : ""} ${
              mechanic.active ? "online" : ""
            }`}
          >
            <i>{mechanic.label}</i>
            <b>{mechanic.value}</b>
          </span>
        ))}
      </div>

      <div className="leader-zone">
        <div className="leader-zone-label">
          <span>主战者区域</span>
          <small>{player.leader_area_used}/{player.leader_area_limit}</small>
        </div>
        <div className="leader-zone-slots">
          {Array.from({ length: player.leader_area_limit }).map((_, index) => {
            const content = leaderContents[index];
            return content ? (
              <span
                key={content.entity_id}
                className={`leader-zone-slot filled ${content.kind}`}
                title={`${content.value}：${content.name}`}
              >
                <i
                  className="leader-zone-art"
                  style={
                    content.image_url
                      ? { backgroundImage: `url("${imageUrl(content.image_url)}")` }
                      : undefined
                  }
                />
                <b>{content.value}</b>
                <small>{content.name}</small>
              </span>
            ) : (
              <span className="leader-zone-slot" key={`empty-${index}`}>
                <i>{index + 1}</i>
                <small>空</small>
              </span>
            );
          })}
        </div>
      </div>
    </section>
  );
}

function BoardRow({
  cards,
  actions,
  selected,
  onSelect,
}: {
  cards: CardView[];
  actions: MatchAction[];
  selected: number | null;
  onSelect: (card: CardView) => void;
}) {
  return (
    <div className="board-row">
      {cards.length === 0 ? (
        <div className="empty-board">场上没有卡牌</div>
      ) : (
        cards.map((card) => {
          const actionable = actions.some(
            (action) =>
              action.source_entity_id === card.entity_id ||
              action.target_entity_id === card.entity_id,
          );
          return (
            <CardTile
              key={card.entity_id}
              card={card}
              compact
              selected={selected === card.entity_id}
              actionable={actionable}
              onClick={() => onSelect(card)}
            />
          );
        })
      )}
    </div>
  );
}

function AnimationStage({
  queue,
  index,
  playing,
  onReplay,
  onSkip,
}: {
  queue: AnimationCue[];
  index: number;
  playing: boolean;
  onReplay: () => void;
  onSkip: () => void;
}) {
  const cue = queue[index];
  if (!cue) {
    return (
      <section className="battle-broadcast idle">
        <div>
          <span className="eyebrow">RESOLUTION</span>
          <strong>等待下一次结算</strong>
        </div>
        <small>攻击、法术、护符与效果会显示在这里</small>
      </section>
    );
  }
  return (
    <section className={`battle-broadcast cue-${cue.kind} ${playing ? "playing" : ""}`}>
      <div className="broadcast-heading">
        <div>
          <span className="eyebrow">RESOLUTION · {index + 1}/{queue.length}</span>
          <strong key={cue.id}>{cue.title}</strong>
        </div>
        <span className="cue-kind">{cue.kind.replaceAll("_", " ")}</span>
      </div>
      {cue.detail && <p>{cue.detail}</p>}
      <div className="broadcast-controls">
        <button type="button" onClick={onReplay}>重播</button>
        {playing && <button type="button" onClick={onSkip}>跳过</button>}
      </div>
    </section>
  );
}

function HistoryDrawer({
  open,
  summaries,
  selected,
  loading,
  onClose,
  onSelect,
}: {
  open: boolean;
  summaries: HistorySummary[];
  selected: HistoryRecord | null;
  loading: boolean;
  onClose: () => void;
  onSelect: (matchId: string) => void;
}) {
  if (!open) return null;
  return (
    <div className="history-backdrop" role="presentation">
      <aside className="history-drawer" aria-label="持久化对局记录">
        <header>
          <div>
            <span className="eyebrow">LOCAL MATCH ARCHIVE</span>
            <h2>对局记录</h2>
          </div>
          <button type="button" onClick={onClose} aria-label="关闭对局记录">关闭</button>
        </header>
        <div className="history-layout">
          <div className="history-list">
            {loading && summaries.length === 0 && <p className="history-empty">读取记录中…</p>}
            {!loading && summaries.length === 0 && <p className="history-empty">还没有保存的对局</p>}
            {summaries.map((summary) => (
              <button
                type="button"
                key={summary.match_id}
                className={selected?.match_id === summary.match_id ? "selected" : ""}
                onClick={() => onSelect(summary.match_id)}
              >
                <span>
                  <b>{historyStatus(summary)}</b>
                  <small>{formatHistoryTime(summary.created_at)}</small>
                </span>
                <span>
                  <i>Turn {summary.turn ?? "?"}</i>
                  <i>{summary.action_count ?? 0} 步</i>
                  <i>Seed {summary.seed ?? "?"}</i>
                </span>
              </button>
            ))}
          </div>
          <div className="history-detail">
            {!selected ? (
              <div className="history-empty">选择一局查看完整操作与结算时间线</div>
            ) : (
              <>
                <div className="history-summary">
                  <span className={`history-result status-${selected.status}`}>
                    {historyStatus(selected)}
                  </span>
                  <div>
                    <b>{selected.deck.display_name}</b>
                    <small>
                      {formatHistoryTime(selected.created_at)} · Seed {selected.seed} · Turn {selected.turn}
                    </small>
                  </div>
                </div>
                <div className="history-actions">
                  {selected.actions.map((action) => {
                    const revealPrivate = historyRevealsPrivateInformation(selected);
                    const decisionActions = [...(action.decision?.legal_actions ?? [])]
                      .sort(
                        (left, right) =>
                          (right.probability ?? -1) - (left.probability ?? -1),
                      );
                    const aiHand = action.before?.players[action.player_index]?.hand;
                    return (
                      <article key={action.sequence}>
                        <div className="history-action-heading">
                          <span>#{action.sequence}</span>
                          <b>
                            {action.actor_role === "human" ? "你" : "AI"}：
                            {historyActionLabel(selected, action)}
                          </b>
                        </div>
                        {action.animations.map((rawCue) => {
                          const cue = historyCue(selected, rawCue);
                          return (
                            <p key={cue.id} className={`history-cue cue-${cue.kind}`}>
                              <i>{cue.kind}</i>
                              <span>{cue.title}</span>
                              {cue.detail && <small>{cue.detail}</small>}
                            </p>
                          );
                        })}
                        {revealPrivate && action.actor_role === "ai" && action.decision && (
                          <details className="history-decision">
                            <summary>
                              AI 决策分布 · 估值{" "}
                              {action.decision.value?.toFixed(3) ?? "—"} · 选中概率{" "}
                              {formatPolicyProbability(
                                action.decision.selected_probability,
                              )}
                            </summary>
                            <div className="history-ai-hand">
                              <b>当时 AI 手牌</b>
                              <span>
                                {aiHand?.length
                                  ? aiHand
                                      .map((card) => `${card.name}（${card.cost}）`)
                                      .join("、")
                                  : "无手牌或旧记录未保存"}
                              </span>
                            </div>
                            <div className="history-policy-actions">
                              {decisionActions.map((candidate) => (
                                <p
                                  key={candidate.id}
                                  className={candidate.selected ? "selected" : ""}
                                >
                                  <span>{candidate.label}</span>
                                  <b>
                                    {formatPolicyProbability(
                                      candidate.probability,
                                      2,
                                    )}
                                  </b>
                                  <small>
                                    action #{candidate.id} · logit{" "}
                                    {candidate.logit?.toFixed(4) ?? "—"}
                                  </small>
                                </p>
                              ))}
                            </div>
                          </details>
                        )}
                        {action.logs.length > 0 && (
                          <details className="history-raw-logs">
                            <summary>原始结算日志 · {action.logs.length} 条</summary>
                            {action.logs.map((line, index) => (
                              <p key={`${action.sequence}-log-${index}`}>
                                {historyLog(selected, line)}
                              </p>
                            ))}
                          </details>
                        )}
                      </article>
                    );
                  })}
                  {selected.actions.length === 0 && (
                    <div className="history-empty">这局尚未执行任何动作</div>
                  )}
                </div>
              </>
            )}
          </div>
        </div>
      </aside>
    </div>
  );
}

export default function Home() {
  const [state, setState] = useState<MatchState | null>(null);
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState("");
  const [seed, setSeed] = useState("");
  const [humanDeck, setHumanDeck] = useState("");
  const [aiDeck, setAiDeck] = useState("");
  const [selectedModel, setSelectedModel] = useState("");
  const [selectedEntity, setSelectedEntity] = useState<number | null>(null);
  const [showLog, setShowLog] = useState(false);
  const [animationQueue, setAnimationQueue] = useState<AnimationCue[]>([]);
  const [animationIndex, setAnimationIndex] = useState(0);
  const [animationPlaying, setAnimationPlaying] = useState(false);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historySummaries, setHistorySummaries] = useState<HistorySummary[]>([]);
  const [selectedHistory, setSelectedHistory] = useState<HistoryRecord | null>(null);

  const acceptState = useCallback((next: MatchState, animate: boolean) => {
    setState(next);
    setSeed(String(next.seed));
    setHumanDeck(next.human_deck.name);
    setAiDeck(next.ai_deck.name);
    setSelectedModel(
      next.model?.id ??
        next.available_models?.[0]?.id ??
        next.checkpoint,
    );
    setSelectedEntity(null);
    if (animate && next.animation_batch.length > 0) {
      setAnimationQueue(next.animation_batch);
      setAnimationIndex(0);
      setAnimationPlaying(true);
    } else {
      setAnimationQueue([]);
      setAnimationIndex(0);
      setAnimationPlaying(false);
    }
  }, []);

  const startMatch = useCallback(async () => {
    setBusy(true);
    setError("");
    try {
      const parsedSeed = seed.trim() === "" ? null : Number(seed);
      if (parsedSeed != null && !Number.isInteger(parsedSeed)) {
        throw new Error("种子必须是整数");
      }
      const next = await request<MatchState>("/api/new-match", {
        method: "POST",
        body: JSON.stringify({
          seed: parsedSeed,
          human_player: 0,
          human_deck: humanDeck || undefined,
          ai_deck: aiDeck || undefined,
          model: selectedModel || undefined,
        }),
      });
      acceptState(next, false);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(false);
    }
  }, [acceptState, aiDeck, humanDeck, seed, selectedModel]);

  useEffect(() => {
    let active = true;
    request<MatchState>("/api/state")
      .catch((caught) => {
        if (caught instanceof RequestError && caught.status === 409) {
          return request<MatchState>("/api/new-match", {
            method: "POST",
            body: JSON.stringify({ seed: null, human_player: 0 }),
          });
        }
        throw caught;
      })
      .then((next) => {
        if (active) acceptState(next, false);
      })
      .catch((caught) => {
        if (active) setError(caught instanceof Error ? caught.message : String(caught));
      })
      .finally(() => {
        if (active) setBusy(false);
      });
    return () => {
      active = false;
    };
  }, [acceptState]);

  const currentCue = animationQueue[animationIndex];
  useEffect(() => {
    if (!animationPlaying || !currentCue) return;
    const timer = window.setTimeout(() => {
      if (animationIndex + 1 >= animationQueue.length) {
        setAnimationPlaying(false);
      } else {
        setAnimationIndex((current) => current + 1);
      }
    }, Math.max(450, currentCue.duration_ms));
    return () => window.clearTimeout(timer);
  }, [animationIndex, animationPlaying, animationQueue.length, currentCue]);

  const performAction = useCallback(async (action: number) => {
    setBusy(true);
    setError("");
    try {
      const next = await request<MatchState>("/api/action", {
        method: "POST",
        body: JSON.stringify({ action }),
      });
      acceptState(next, true);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(false);
    }
  }, [acceptState]);

  const openHistory = useCallback(async () => {
    setHistoryOpen(true);
    setHistoryLoading(true);
    setSelectedHistory(null);
    try {
      const payload = await request<{ matches: HistorySummary[] }>("/api/history");
      setHistorySummaries(payload.matches);
      const current = state?.match_id;
      if (current && payload.matches.some((item) => item.match_id === current)) {
        const record = await request<HistoryRecord>(`/api/history/${current}`);
        setSelectedHistory(record);
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setHistoryLoading(false);
    }
  }, [state]);

  const selectHistory = useCallback(async (matchId: string) => {
    setHistoryLoading(true);
    try {
      const record = await request<HistoryRecord>(`/api/history/${matchId}`);
      setSelectedHistory(record);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setHistoryLoading(false);
    }
  }, []);

  const visibleActions = useMemo(() => {
    if (!state) return [];
    if (state.pending_choice || selectedEntity == null) return state.actions;
    const scoped = state.actions.filter(
      (action) =>
        action.source_entity_id === selectedEntity ||
        action.target_entity_id === selectedEntity,
    );
    return scoped.length > 0 ? scoped : state.actions;
  }, [selectedEntity, state]);

  const interactionLocked = busy || animationPlaying;
  const handleCardSelect = (card: CardView) => {
    if (!state || interactionLocked) return;
    const targetActions = state.actions.filter(
      (action) => action.target_entity_id === card.entity_id,
    );
    if (state.pending_choice && targetActions.length === 1) {
      void performAction(targetActions[0].id);
      return;
    }
    setSelectedEntity((current) =>
      current === card.entity_id ? null : card.entity_id,
    );
  };

  if (!state) {
    return (
      <main className="loading-screen">
        <div className="loader-mark">SWB</div>
        <h1>正在载入对局引擎</h1>
        <p>读取卡组、规则、历史记录和 PPO 策略…</p>
        {error && (
          <div className="connection-error">
            <strong>无法连接本地引擎</strong>
            <span>{error}</span>
            <button type="button" onClick={() => void startMatch()}>重试</button>
          </div>
        )}
      </main>
    );
  }

  const human = state.players[state.human_player];
  const ai = state.players[state.ai_player];
  const activeModel = state.model ?? {
    id: state.checkpoint,
    display_name: state.checkpoint,
    group: "当前模型",
    filename: state.checkpoint,
    size_bytes: 0,
  };
  const availableModels = state.available_models ?? [activeModel];
  const outcome = state.terminated
    ? state.winner === state.human_player
      ? "你获胜了"
      : state.winner == null
        ? "对局结束"
        : "AI 获胜"
    : state.truncated
      ? "对局达到步数上限"
      : null;
  const statusText = state.human_turn
    ? state.pending_choice?.prompt || "轮到你行动"
    : "AI 正在行动";

  return (
    <main className="simulator-shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">SWB</span>
          <span>
            <strong>对局模拟器</strong>
            <small>Human vs PPO · 本地推理</small>
          </span>
        </div>
        <div className="match-meta">
          <span>你：{state.human_deck.display_name}</span>
          <span>AI：{state.ai_deck.display_name}</span>
          <span>模型：{activeModel.display_name}</span>
          <span>Turn {state.turn}</span>
          <span>Seed {state.seed}</span>
        </div>
        <div className="new-match">
          <button type="button" className="history-button" onClick={() => void openHistory()}>
            对局记录
          </button>
          <label className="deck-picker">
            <small>AI 模型</small>
            <select
              value={selectedModel}
              onChange={(event) => setSelectedModel(event.target.value)}
              aria-label="选择 AI 模型"
              disabled={busy}
            >
              {availableModels.map((model) => (
                <option value={model.id} key={model.id}>
                  {model.display_name}
                </option>
              ))}
            </select>
          </label>
          <label className="deck-picker">
            <small>我的卡组</small>
            <select
              value={humanDeck}
              onChange={(event) => setHumanDeck(event.target.value)}
              aria-label="选择我的卡组"
              disabled={busy}
            >
              {state.available_decks.map((deck) => (
                <option value={deck.name} key={deck.name}>
                  {deck.display_name}
                </option>
              ))}
            </select>
          </label>
          <label className="deck-picker">
            <small>AI 卡组</small>
            <select
              value={aiDeck}
              onChange={(event) => setAiDeck(event.target.value)}
              aria-label="选择 AI 卡组"
              disabled={busy}
            >
              {state.available_decks.map((deck) => (
                <option value={deck.name} key={deck.name}>
                  {deck.display_name}
                  {deck.name === state.specialist_deck?.name ? " · 专精" : ""}
                </option>
              ))}
            </select>
          </label>
          <input
            value={seed}
            onChange={(event) => setSeed(event.target.value)}
            placeholder="随机种子"
            aria-label="对局随机种子"
          />
          <button type="button" onClick={() => void startMatch()} disabled={busy}>
            新对局
          </button>
        </div>
      </header>

      {state.warnings.map((warning) => (
        <div className="warning-banner" key={warning}>
          <strong>兼容性提示</strong>
          <span>{warning.split(" checkpoint=")[0]}</span>
        </div>
      ))}
      {error && <div className="error-banner">{error}</div>}

      <div className="game-layout">
        <section className="battlefield">
          <div className="opponent-hand" aria-label={`AI 手牌 ${ai.hand_count} 张`}>
            {Array.from({ length: ai.hand_count }).map((_, index) => (
              <span className="card-back" key={index}>SWB</span>
            ))}
          </div>

          <LeaderPanel
            player={ai}
            active={state.current_player === state.ai_player}
            first={state.first_player === state.ai_player}
            label="AI"
          />

          <BoardRow
            cards={ai.board}
            actions={state.actions}
            selected={selectedEntity}
            onSelect={handleCardSelect}
          />

          <div className="turn-divider">
            <span className={state.human_turn ? "human-ready" : ""}>
              {busy ? "结算中…" : animationPlaying ? "播放结算…" : statusText}
            </span>
            <small>{state.phase === "mulligan" ? "起手换牌" : `第 ${state.turn} 回合`}</small>
          </div>

          <BoardRow
            cards={human.board}
            actions={state.actions}
            selected={selectedEntity}
            onSelect={handleCardSelect}
          />

          <LeaderPanel
            player={human}
            active={state.current_player === state.human_player}
            first={state.first_player === state.human_player}
            label="你"
          />

          <div className="human-hand">
            {human.hand?.map((card) => {
              const actionable = state.actions.some(
                (action) => action.source_entity_id === card.entity_id,
              );
              return (
                <CardTile
                  key={card.entity_id}
                  card={card}
                  selected={selectedEntity === card.entity_id}
                  actionable={actionable}
                  onClick={() => handleCardSelect(card)}
                />
              );
            })}
          </div>
        </section>

        <aside className="control-rail">
          <AnimationStage
            queue={animationQueue}
            index={animationIndex}
            playing={animationPlaying}
            onReplay={() => {
              setAnimationIndex(0);
              setAnimationPlaying(animationQueue.length > 0);
            }}
            onSkip={() => {
              setAnimationIndex(Math.max(0, animationQueue.length - 1));
              setAnimationPlaying(false);
            }}
          />

          <section className="action-panel">
            <div className="panel-heading">
              <div>
                <span className="eyebrow">LEGAL ACTIONS</span>
                <h2>{state.pending_choice?.prompt || "可用操作"}</h2>
              </div>
              <span className="action-count">{visibleActions.length}</span>
            </div>
            {visibleActions.length === 0 ? (
              <div className="no-actions">
                {outcome || (busy ? "AI 思考中…" : "请选择一张卡牌")}
              </div>
            ) : (
              <div className="action-list">
                {visibleActions.map((action) => (
                  <button
                    key={action.id}
                    type="button"
                    disabled={interactionLocked}
                    onClick={() => void performAction(action.id)}
                    className={`action-button action-${action.kind}`}
                  >
                    <span>{action.label}</span>
                    <small>#{action.id}</small>
                  </button>
                ))}
              </div>
            )}
          </section>

          {state.last_ai_actions.length > 0 && (
            <section className="ai-summary">
              <span className="eyebrow">AI LAST TURN</span>
              {state.last_ai_actions.slice(-4).map((action, index) => (
                <p key={`${action}-${index}`}>{action}</p>
              ))}
            </section>
          )}

          <section className="log-panel">
            <button
              type="button"
              className="log-toggle"
              onClick={() => setShowLog((visible) => !visible)}
            >
              <span>原始对局日志</span>
              <small>{showLog ? "收起" : "展开"} · {state.logs.length}</small>
            </button>
            {showLog && (
              <div className="log-list">
                {[...state.logs].reverse().map((line, index) => (
                  <p key={`${line}-${index}`}>{line}</p>
                ))}
              </div>
            )}
          </section>

          <footer className="runtime-note">
            <span>{state.checkpoint}</span>
            <span>记录 {state.match_id.slice(-8)}</span>
          </footer>
        </aside>
      </div>

      {outcome && !animationPlaying && (
        <div className="result-overlay">
          <div className="result-card">
            <span className="eyebrow">MATCH COMPLETE</span>
            <h2>{outcome}</h2>
            <p>第 {state.turn} 回合 · Seed {state.seed}</p>
            <div className="result-actions">
              <button type="button" onClick={() => void openHistory()}>查看本局记录</button>
              <button type="button" onClick={() => void startMatch()}>再来一局</button>
            </div>
          </div>
        </div>
      )}

      <HistoryDrawer
        open={historyOpen}
        summaries={historySummaries}
        selected={selectedHistory}
        loading={historyLoading}
        onClose={() => setHistoryOpen(false)}
        onSelect={(matchId) => void selectHistory(matchId)}
      />
    </main>
  );
}
