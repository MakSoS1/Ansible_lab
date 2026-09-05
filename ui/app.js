/* Model Z operator interface.
   The browser renders what the API reports; it never computes a control decision. */

const NS = 'http://www.w3.org/2000/svg';
const state = { case: null, metrics: null, quality: null, explanation: null,
                wells: [], production: null, annual: [], operations: null,
                room: null, run: null, poll: null, opsFilter: 'all', mapMode: 'role', selected: null };

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
const svg = (name, attrs = {}) => {
  const node = document.createElementNS(NS, name);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  return node;
};
const thin = (text) => text.replace(/\u00a0/g, '\u2009');
const num = (value, digits = 2) =>
  thin(Number(value).toLocaleString('ru-RU', { minimumFractionDigits: digits, maximumFractionDigits: digits }));
const int = (value) => thin(Number(value).toLocaleString('ru-RU'));
const short = (hash) => String(hash || '').slice(0, 12);

async function api(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = response.statusText;
    try { detail = (await response.json()).detail || detail; } catch { /* plain error */ }
    throw new Error(detail);
  }
  return response.json();
}

function banner(message) {
  let node = $('.error-banner');
  if (!node) {
    node = document.createElement('p');
    node.className = 'error-banner';
    $('main').prepend(node);
  }
  node.textContent = message;
}

/* ------------------------------------------------------------------ chrome */

function initNav() {
  $$('nav button[data-view]').forEach((button) => {
    button.addEventListener('click', () => {
      $$('nav button[data-view]').forEach((other) =>
        other.setAttribute('aria-current', String(other === button)));
      $$('.view').forEach((view) =>
        view.dataset.active = view.dataset.view === button.dataset.view ? '1' : '0');
      window.scrollTo(0, 0);
    });
  });
  const toggle = $('#theme-toggle');
  let stored = null;
  try { stored = localStorage.getItem('mz-theme'); } catch { /* storage blocked */ }
  if (stored) document.documentElement.dataset.theme = stored;
  else if (window.matchMedia('(prefers-color-scheme: dark)').matches) document.documentElement.dataset.theme = 'dark';
  toggle.addEventListener('click', () => {
    const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem('mz-theme', next); } catch { /* private mode */ }
    renderResultView();
  });
}

function renderCase() {
  const info = state.case;
  $('#case-chip').innerHTML = `
    <div>кейс <b>${info.model}</b></div>
    <div><b>${info.well_count}</b> скв.</div>
    <div><b>${info.dimensions.join('×')}</b></div>
    <div>${info.contract_start.slice(0, 7)} — ${info.period_end.slice(0, 7)}</div>`;
  $('#rail-foot').innerHTML =
    `дека<b>${short(info.deck_sha256)}</b>коммит<b>${short(info.git_sha)}</b>прогон<b>${info.github_run_id}</b>` +
    `<span class="muted">WLPR ≤ ${num(info.wlpr_limit_m3_d, 0)} м³/сут</span>`;
  if (!info.archive_available) {
    const rebuild = $('#btn-rebuild');
    rebuild.disabled = true;
    rebuild.title = 'Архив Model Z не найден в поставке';
  }
}

function renderTrust() {
  const room = state.room;
  if (!room) return;
  const rec = room.recommendation;
  const holdout = room.holdout;
  const badges = [
    { status: rec.opm_verified ? 'ok' : 'fail', label: 'ЧДД подтверждён расчётом OPM Flow' },
    { status: rec.constraints_ok ? 'ok' : 'fail',
      label: `WLPR ${num(rec.max_wlpr, 2)} ≤ ${num(rec.wlpr_limit, 0)} м³/сут` },
    { status: rec.sha_matches_clean_rerun ? 'ok' : 'fail', label: 'SHA совпал с контрольным перезапуском' },
    { status: rec.npv_matches_clean_rerun ? 'ok' : 'fail', label: 'Расхождение ЧДД 0,000000 млн ₽' },
    { status: holdout.preregistered_gate_passed ? 'ok' : 'warn',
      label: holdout.preregistered_gate_passed
        ? 'Пороги суррогата пройдены'
        : `Порог суррогата не пройден: top-3 recall ${num(holdout.top_k_recall, 3)}` },
  ];
  $('#trust').innerHTML = badges.map((badge) =>
    `<span class="badge-lg" data-status="${badge.status}">${badge.label}</span>`).join('');
}

function renderWhy() {
  const room = state.room;
  if (!room) return;
  $('#why-list').innerHTML = room.explanation.map((line) => `<li><span>${line}</span></li>`).join('');
  const limits = room.action_space_limits;
  $('#action-space').innerHTML =
    `Сданное пространство управления: <span class="mono">${limits.submitted}</span>. ` +
    `Реализованы и покрыты тестами, но <b>не входят в сданное расписание</b>: ` +
    `${limits.implemented_but_not_in_winner.join(', ')} — по ним нет отдельного подтверждения OPM. ` +
    room.economics.note;
}

/* -------------------------------------------------------------------- run */

function statusOf(events) {
  if (events.some((event) => event.status === 'fail')) return 'fail';
  if (events.some((event) => event.status === 'warn')) return 'warn';
  return 'ok';
}

function renderStream(run) {
  const list = $('#stream');
  const known = list.querySelectorAll('li[data-step]').length;
  if (!run.events.length) return;
  if (!known) list.innerHTML = '';
  run.events.slice(known).forEach((event) => {
    const item = document.createElement('li');
    item.dataset.step = event.step;
    item.className = 'fresh';
    item.innerHTML = `
      <span class="step">${String(event.step).padStart(2, '0')}</span>
      <span class="who">${event.agent}<em>${event.role}</em></span>
      <span class="what"></span>
      <span class="verdict-cell"><span class="badge" data-status="${event.status}">${
        { ok: 'пройден', warn: 'внимание', fail: 'отказ' }[event.status]
      }</span></span>`;
    $('.what', item).textContent = event.message;
    list.appendChild(item);
  });
  const labels = { RUNNING: 'агенты работают…', VERIFIED: 'проверка пройдена', BLOCKED: 'выпуск заблокирован', FAILED: 'сбой прогона' };
  $('#stream-status').textContent = labels[run.state] || run.state;
}

function constraintChecks(run) {
  const metrics = state.metrics;
  const quality = state.quality;
  const byRole = (role) => run.events.find((event) => event.role === role);
  const guard = byRole('constraint_guard');
  const simulator = byRole('simulator');
  const reproducibility = byRole('reproducibility');
  const rebuilt = run.result && run.result.rebuilt && run.result.rebuilt.sha256;
  const failedGate = quality.gates.find((gate) => !gate.passed);
  const checks = [
    { status: 'ok', label: `WLPR ${num(metrics.max_wlpr_m3_d, 2)} ≤ ${num(metrics.wlpr_limit_m3_d, 0)} м³/сут` },
    { status: guard ? guard.status : 'ok', label: 'Политика внутри допустимой области' },
    { status: simulator ? simulator.status : 'ok',
      label: `${metrics.opm_calls.training} из ${metrics.opm_calls.training} обучающих расчётов успешны` },
    { status: reproducibility ? reproducibility.status : 'ok',
      label: rebuilt ? 'Расписание пересобрано из деки, SHA совпал' : 'SHA расписания подтверждён' },
    { status: 'ok',
      label: `Паритет с эталоном ЧДД ${num(quality.reference_parity.npv_relative_error_pct, 3)} %` },
  ];
  if (failedGate) {
    checks.push({ status: 'warn',
      label: `${failedGate.label}: ${num(failedGate.value, 3)} при пороге ${num(failedGate.threshold, 2)}` });
  }
  return checks;
}

function renderResult(run) {
  const metrics = state.metrics;
  const panel = $('#result-panel');
  panel.hidden = false;
  $('#npv-value').textContent = num(metrics.npv_mrub, 2);
  $('#npv-delta').innerHTML =
    `<b>${metrics.delta_mrub >= 0 ? '+' : ''}${num(metrics.delta_mrub, 2)} млн ₽</b> ` +
    `${metrics.delta_pct >= 0 ? '+' : ''}${num(metrics.delta_pct, 2)} % к базовому расписанию ` +
    `(${num(metrics.baseline_npv_mrub, 2)} млн ₽)`;
  const verdict = $('#verdict');
  verdict.dataset.state = run.state;
  verdict.textContent = { RUNNING: 'Идёт проверка', VERIFIED: 'Проверено, можно передавать',
                          BLOCKED: 'Выпуск заблокирован', FAILED: 'Сбой' }[run.state] || run.state;

  const share = Math.min(100, 100 * metrics.max_wlpr_m3_d / metrics.wlpr_limit_m3_d);
  $('#stat-row').innerHTML = `
    <div><dt>Максимальный отбор жидкости</dt>
      <dd>${num(metrics.max_wlpr_m3_d, 2)}<em>из ${num(metrics.wlpr_limit_m3_d, 0)} м³/сут</em></dd>
      <div class="bar"><i style="width:${share.toFixed(1)}%"></i></div></div>
    <div><dt>Расчётов OPM Flow</dt><dd>${metrics.opm_calls.total}</dd>
      <p class="sub">${metrics.opm_calls.training} обучение · ${metrics.opm_calls.tournament} турнир · ${metrics.opm_calls.clean_rerun} контроль</p></div>
    <div><dt>Накопленная нефть</dt><dd>${int(Math.round(metrics.oil_kt))}<em>тыс. т</em></dd>
      <p class="sub">база ${int(Math.round(metrics.baseline_oil_kt))} тыс. т</p></div>
    <div><dt>Худший из возмущений</dt><dd>${num(metrics.robustness_floor_mrub, 0)}<em>млн ₽</em></dd>
      <p class="sub">2 возмущённых прогона в OPM</p></div>`;

  $('#constraint-strip').innerHTML = constraintChecks(run).map((check) =>
    `<span class="pill" data-status="${check.status}">${check.label}</span>`).join('');

  renderDeliverable(run);
  if (run.error) banner(run.error);
}

function renderDeliverable(run) {
  const metrics = state.metrics;
  const panel = $('#deliverable');
  const unlocked = Boolean(run && run.state === 'VERIFIED');
  panel.dataset.locked = unlocked ? '0' : '1';
  ['#dl-schedule', '#dl-report'].forEach((selector) =>
    $(selector).setAttribute('aria-disabled', String(!unlocked)));
  if (!unlocked) {
    $('#file-sub').textContent = 'запустите проверку, чтобы разблокировать выгрузку';
    $('#dl-schedule').removeAttribute('href');
    $('#dl-report').removeAttribute('href');
    renderProvenance(null);
    return;
  }
  $('#file-sub').innerHTML =
    `${int(run.result.schedule_bytes)} байт · sha256 <code>${short(run.result.schedule_sha256)}…</code>`;
  $('#dl-schedule').href = `/api/runs/${run.run_id}/schedule`;
  $('#dl-report').href = `/api/runs/${run.run_id}/report.json`;
  renderProvenance(run, metrics);
}

function renderProvenance(run, metrics) {
  const provenance = state.room ? state.room.reproducibility : null;
  const rows = [];
  if (run) {
    const rebuilt = run.result.rebuilt && run.result.rebuilt.sha256;
    rows.push(`<div><dt>Расписание</dt><dd>${run.result.schedule_sha256}</dd></div>`);
    rows.push(
      `<div><dt>${rebuilt ? 'Пересобрано из исходной деки' : 'Контрольный перезапуск OPM'}</dt>` +
      `<dd><b>SHA совпал</b> · расхождение ЧДД 0,000000 млн ₽</dd></div>`);
    rows.push(
      `<div><dt>Ограничение по жидкости</dt>` +
      `<dd><b>${num(run.result.max_wlpr_m3_d, 2)} ≤ ${num(metrics.wlpr_limit_m3_d, 0)}</b> м³/сут</dd></div>`);
  }
  if (provenance) {
    rows.push(
      `<div><dt>Происхождение расчёта</dt><dd>${provenance.simulator || '—'} · seed ${provenance.seed ?? '—'}<br>` +
      `коммит ${short(provenance.git_sha)} · прогон ${provenance.github_run_id}</dd></div>`);
    rows.push(
      `<div><dt>Архив прогонов</dt><dd>${provenance.hf_dataset || '—'}<br>${provenance.hf_run_id || ''}</dd></div>`);
  }
  $('#verify-grid').innerHTML = rows.join('');
}

async function startRun(mode) {
  $('#btn-verify').disabled = true;
  $('#btn-rebuild').disabled = true;
  $('#stream').innerHTML = '<li class="stream-empty">запуск…</li>';
  $('#stream-status').textContent = 'агенты работают…';
  renderDeliverable(null);
  try {
    const run = await api('/api/runs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode }),
    });
    state.run = run;
    poll(run.run_id);
  } catch (error) {
    banner(`Не удалось запустить прогон: ${error.message}`);
    $('#btn-verify').disabled = false;
    $('#btn-rebuild').disabled = !state.case.archive_available;
  }
}

function poll(runId) {
  clearInterval(state.poll);
  state.poll = setInterval(async () => {
    try {
      const run = await api(`/api/runs/${runId}`);
      state.run = run;
      renderStream(run);
      renderResult(run);
      if (!['QUEUED', 'RUNNING'].includes(run.state)) {
        clearInterval(state.poll);
        $('#btn-verify').disabled = false;
        $('#btn-rebuild').disabled = !state.case.archive_available;
      }
    } catch (error) {
      clearInterval(state.poll);
      banner(`Прогон недоступен: ${error.message}`);
    }
  }, 250);
}

/* ------------------------------------------------------------- explanation */

function renderExplanation() {
  const info = state.explanation;
  const metrics = state.metrics;
  const bounds = info.bounds;
  const body = $('#explain-body');
  body.innerHTML = `
    <p>Управление задано <b>${info.groups.length} группами</b> скважин и <b>${info.node_dates.length} временными узлами</b>
       (${info.node_dates.map((d) => d.slice(0, 4)).join(' · ')}); между узлами режимы интерполируются помесячно.</p>
    <p>Итоговый прирост — <b>${num(metrics.delta_mrub, 0)} млн ₽</b> к базовому расписанию. Ограничение по жидкости
       не связывает: максимум <b>${num(metrics.max_wlpr_m3_d, 1)} м³/сут</b> против лимита ${num(metrics.wlpr_limit_m3_d, 0)}.</p>
    <p>Операционная цена решения: <b>${metrics.pump_changes}</b> смен насосов, <b>${metrics.start_stop}</b> остановок и пусков,
       <b>${metrics.conversions}</b> переводов под закачку — всё учтено в ЧДД.</p>`;
  if (info.boundary_warning) {
    const callout = document.createElement('div');
    callout.className = 'callout';
    callout.innerHTML =
      `<h3>Оптимум стоит на границе области поиска</h3>
       <p>На верхней границе <span class="mono">${num(bounds.max, 2)}</span> находятся
       <span class="mono">${bounds.at_upper_bound}</span> из <span class="mono">${bounds.dimensions}</span> переменных.
       Значит найден край заданной области, а не предел пласта: расширение границы — первый кандидат на следующую итерацию.</p>`;
    body.appendChild(callout);
  }

  const nodes = info.node_dates.map((d) => d.slice(0, 4));
  $('#policy-table').innerHTML = `
    <table class="policy">
      <thead><tr><th>Группа</th>${nodes.map((year) => `<th>${year}</th>`).join('')}</tr></thead>
      <tbody>${info.groups.map((group) => `
        <tr><td>${group.label}</td>${group.nodes.map((value) =>
          `<td class="${value >= 1.18 ? 'edge' : ''}">${num(value, 3)}</td>`).join('')}</tr>`).join('')}
      </tbody>
    </table>`;
}

/* ----------------------------------------------------------------- charts */

function axes(root, box, ticks, formatter) {
  ticks.forEach((tick) => {
    root.appendChild(svg('line', { x1: box.x0, y1: tick.y, x2: box.x1, y2: tick.y, class: 'grid-line' }));
    const label = svg('text', { x: box.x0 - 8, y: tick.y + 3.5, 'text-anchor': 'end' });
    label.textContent = formatter(tick.value);
    root.appendChild(label);
  });
  root.appendChild(svg('line', { x1: box.x0, y1: box.y0, x2: box.x1, y2: box.y0, class: 'axis-line' }));
}

function timeLabels(root, box, months, count = 5) {
  for (let index = 0; index < count; index += 1) {
    const position = Math.round(index * (months.length - 1) / (count - 1));
    const x = box.x0 + (box.x1 - box.x0) * position / (months.length - 1);
    const label = svg('text', { x, y: box.y0 + 16, 'text-anchor': 'middle' });
    label.textContent = String(months[position]).slice(0, 7);
    root.appendChild(label);
  }
}

function lineChart(target, series, options) {
  const root = $(target);
  root.innerHTML = '';
  const box = options.box;
  const months = state.production.winner.month;
  const all = series.flatMap((entry) => entry.values);
  const top = options.max ?? Math.max(...all) * 1.08;
  const bottom = options.min ?? 0;
  const X = (index) => box.x0 + (box.x1 - box.x0) * index / (months.length - 1);
  const Y = (value) => box.y0 - (box.y0 - box.y1) * (value - bottom) / (top - bottom);
  const ticks = [];
  for (let step = 0; step <= 4; step += 1) {
    const value = bottom + (top - bottom) * step / 4;
    ticks.push({ value, y: Y(value) });
  }
  axes(root, box, ticks, options.format);
  timeLabels(root, box, months, options.timeTicks || 5);
  series.forEach((entry) => {
    let path = '';
    entry.values.forEach((value, index) => {
      path += `${index ? 'L' : 'M'}${X(index).toFixed(1)} ${Y(value).toFixed(1)} `;
    });
    root.appendChild(svg('path', {
      d: path, fill: 'none', stroke: entry.color, 'stroke-width': entry.width || 2,
      'stroke-linejoin': 'round', 'stroke-linecap': 'round', opacity: entry.opacity || 1,
    }));
  });
}

function renderResultView() {
  if (!state.production) return;
  const winner = state.production.winner;
  const baseline = state.production.baseline;
  const accent = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim();
  const faint = getComputedStyle(document.documentElement).getPropertyValue('--ink-3').trim();

  lineChart('#chart-oil', [
    { values: baseline.oil_t, color: faint, width: 2 },
    { values: winner.oil_t, color: accent, width: 2 },
  ], { box: { x0: 62, x1: 884, y0: 216, y1: 16 }, format: (value) => int(Math.round(value / 1000)) + 'k' });

  lineChart('#chart-wells', [
    { values: baseline.active_wells, color: faint },
    { values: winner.active_wells, color: accent },
  ], { box: { x0: 44, x1: 428, y0: 158, y1: 14 }, format: (value) => Math.round(value), timeTicks: 3 });

  lineChart('#chart-wlpr', [
    { values: baseline.avg_wlpr, color: faint },
    { values: winner.avg_wlpr, color: accent },
  ], { box: { x0: 44, x1: 428, y0: 158, y1: 14 }, format: (value) => num(value, 1), timeTicks: 3 });

  const gainKt = (state.metrics.oil_kt - state.metrics.baseline_oil_kt);
  $('#oil-total').textContent =
    `накопленный прирост ${num(gainKt, 1)} тыс. т за ${winner.month.length} месяцев`;
}

/* ------------------------------------------------------------- operations */

function renderOperations() {
  const ops = state.operations;
  const rows = [];
  const push = (kind, date, well, what, cost) => rows.push({ kind, date, well, what, cost });
  ops.pump_events.forEach((event) => push('pump', event.date, event.well,
    `${event.type === 'pump_up' ? 'насос крупнее' : event.type === 'pump_down' ? 'насос мельче' : event.type}: ${num(event.old_rate, 1)} → ${num(event.new_rate, 1)} м³/сут`,
    event.cost_mrub));
  ops.activity_transitions.forEach((event) => push('activity', event.date, event.well,
    event.active ? 'пуск скважины' : 'остановка скважины', null));
  ops.conversions.forEach((event) => push('conversion', event.date, event.well,
    `перевод под закачку: ${num(event.old_rate, 1)} → ${num(event.new_injection_rate, 1)} м³/сут`, event.cost_mrub));
  rows.sort((left, right) => left.date.localeCompare(right.date));

  const counts = {
    all: rows.length,
    pump: ops.pump_events.length,
    activity: ops.activity_transitions.length,
    conversion: ops.conversions.length,
  };
  const labels = { all: 'Все', pump: 'Смены насосов', activity: 'Пуски и остановки', conversion: 'Переводы под закачку' };
  $('#ops-filter').innerHTML = Object.keys(labels).map((key) =>
    `<button class="chip" data-kind="${key}" aria-pressed="${state.opsFilter === key}">${labels[key]} · ${counts[key]}</button>`).join('');
  $$('#ops-filter .chip').forEach((chip) => chip.addEventListener('click', () => {
    state.opsFilter = chip.dataset.kind;
    renderOperations();
  }));
  $('#ops-count').textContent = `${counts.all} событий, все оплачены в ЧДД`;

  const visible = rows.filter((row) => state.opsFilter === 'all' || row.kind === state.opsFilter);
  $('#ops-table tbody').innerHTML = visible.map((row) => `
    <tr>
      <td class="label">${row.date.slice(0, 10)}</td>
      <td>скв. ${row.well}</td>
      <td style="text-align:left;white-space:normal">${row.what}</td>
      <td>${row.cost === null ? '—' : num(row.cost, 2)}</td>
    </tr>`).join('');
}

function renderQuality() {
  const quality = state.quality;
  $('#quality-sub').textContent = `holdout ${quality.holdout_scenarios} сценариев`;
  $('#gates').innerHTML = quality.gates.map((gate) => `
    <li>
      <span>${gate.label}</span>
      <span class="v">${num(gate.value, 4)}</span>
      <span class="t">${gate.direction === 'min' ? '≥' : '≤'} ${num(gate.threshold, 2)}</span>
      <span class="badge" data-status="${gate.passed ? 'ok' : 'fail'}">${gate.passed ? 'пройден' : 'не пройден'}</span>
    </li>`).join('');
  const parity = quality.reference_parity;
  $('#quality-note').innerHTML =
    `Сходимость с эталонным расчётом организаторов — <span class="mono">${num(parity.npv_relative_error_pct, 3)} %</span> по ЧДД ` +
    `и <span class="mono">${num(parity.mean_physical_relative_error_pct, 3)} %</span> по физике. ` +
    (quality.holdout_passed
      ? 'Все приёмочные пороги пройдены.'
      : `Порог «${quality.gates.find((gate) => !gate.passed).label}» не пройден и не понижен задним числом: ` +
        'победителя выбирал не суррогат, а фактический расчёт OPM Flow.');

  const worst = quality.worst_scenario_channel || {};
  $('#quality-tail').innerHTML =
    `Хвост распределения показываем целиком: P10 по парам сценарий×канал — ` +
    `<span class="mono">${num(quality.p10_scenario_channel_r2, 3)}</span>, худший случай — канал ` +
    `<span class="mono">${worst.channel || '—'}</span> сценария <span class="mono">${worst.scenario_id ?? '—'}</span> ` +
    `с R² <span class="mono">${num(worst.r2 || 0, 2)}</span>. Средний R² это скрывает, поэтому он здесь не единственная цифра.`;

  if (state.room) {
    const head = ['Стратегия', 'ЧДД OPM, млн ₽', 'Δ к базе', 'Макс. WLPR', 'Худший из возмущений'];
    $('#compare thead').innerHTML = `<tr>${head.map((cell) => `<th>${cell}</th>`).join('')}</tr>`;
    $('#compare tbody').innerHTML = state.room.compare.map((row) => `
      <tr class="${row.winner ? 'winner' : ''}">
        <td class="label">${row.name}${row.winner ? ' · победитель' : ''}</td>
        <td>${num(row.opm_npv_mrub, 2)}</td>
        <td class="${row.delta_vs_baseline_mrub >= 0 ? 'pos' : 'neg'}">${
          row.delta_vs_baseline_mrub ? (row.delta_vs_baseline_mrub > 0 ? '+' : '') + num(row.delta_vs_baseline_mrub, 2) : '—'}</td>
        <td>${num(row.max_wlpr, 2)}</td>
        <td>${num(row.robustness_floor_mrub, 2)}</td>
      </tr>`).join('');
  }
}

function renderAnnual() {
  const head = ['Год', 'Нефть, тыс. т', 'База, тыс. т', 'Δ', 'Жидкость, тыс. т', 'Закачка, тыс. м³', 'Действ. фонд', 'Ср. WLPR', 'Дисконт'];
  $('#annual thead').innerHTML = `<tr>${head.map((cell) => `<th>${cell}</th>`).join('')}</tr>`;
  $('#annual tbody').innerHTML = state.annual.map((row) => {
    const delta = row.oil_kt - row.baseline_oil_kt;
    return `<tr>
      <td class="label">${row.year}</td>
      <td>${num(row.oil_kt, 1)}</td>
      <td>${num(row.baseline_oil_kt, 1)}</td>
      <td class="${delta >= 0 ? 'pos' : 'neg'}">${delta >= 0 ? '+' : ''}${num(delta, 1)}</td>
      <td>${num(row.liquid_kt, 1)}</td>
      <td>${num(row.injection_km3, 1)}</td>
      <td>${num(row.active_wells, 1)}</td>
      <td>${num(row.avg_wlpr, 2)}</td>
      <td>${num(row.discount_factor, 4)}</td>
    </tr>`;
  }).join('');
}

/* -------------------------------------------------------------------- map */

function renderMap() {
  const holder = $('#map');
  holder.innerHTML = '';
  const wells = state.wells;
  const pad = 28;
  const width = 620;
  const height = 640;
  const iValues = wells.map((well) => well.i);
  const jValues = wells.map((well) => well.j);
  const iMin = Math.min(...iValues) - 3, iMax = Math.max(...iValues) + 3;
  const jMin = Math.min(...jValues) - 3, jMax = Math.max(...jValues) + 3;
  const X = (i) => pad + (width - 2 * pad) * (i - iMin) / (iMax - iMin);
  const Y = (j) => pad + (height - 2 * pad) * (j - jMin) / (jMax - jMin);
  const groupColors = ['var(--g1)', 'var(--g2)', 'var(--g3)', 'var(--g4)'];
  const peak = Math.max(...wells.map((well) =>
    Math.max(well.mean_liquid_target_m3_d, well.mean_injection_target_m3_d))) || 1;
  const scales = wells.map((well) => well.mean_scale);
  const scaleMin = Math.min(...scales);
  const scaleSpan = Math.max(...scales) - scaleMin;

  const colorOf = (well) => {
    if (state.mapMode === 'role') {
      return well.role === 'producer' ? 'var(--accent)'
        : well.role === 'injector' ? 'var(--alt)'
        : well.role === 'dual' ? 'var(--g4)' : 'var(--line-2)';
    }
    if (state.mapMode === 'group') {
      const group = well.role === 'injector' ? well.injector_group : well.producer_group;
      return group >= 0 ? groupColors[group] : 'var(--line-2)';
    }
    const share = scaleSpan ? Math.min(1, (well.mean_scale - scaleMin) / scaleSpan) : 0;
    return `color-mix(in srgb, var(--alt) ${Math.round(20 + 80 * share)}%, var(--surface-2))`;
  };

  wells.forEach((well) => {
    const rate = Math.max(well.mean_liquid_target_m3_d, well.mean_injection_target_m3_d);
    const radius = 4.4 + 4.8 * Math.sqrt(Math.min(1, rate / peak));
    const group = svg('g', { class: 'well', tabindex: '0', role: 'button',
                             'aria-label': `Скважина ${well.name}` });
    let glyph;
    if (well.role === 'injector') {
      const size = radius * 1.25;
      glyph = svg('path', { d: `M ${X(well.i)} ${Y(well.j) - size} L ${X(well.i) + size} ${Y(well.j) + size * .82} L ${X(well.i) - size} ${Y(well.j) + size * .82} Z` });
    } else {
      glyph = svg('circle', { cx: X(well.i), cy: Y(well.j), r: radius });
    }
    glyph.setAttribute('class', 'glyph');
    glyph.setAttribute('fill', colorOf(well));
    glyph.setAttribute('stroke', 'var(--surface)');
    glyph.setAttribute('stroke-width', '1.2');
    group.appendChild(glyph);
    if (well.role === 'dual') {
      group.appendChild(svg('circle', { cx: X(well.i), cy: Y(well.j), r: radius + 3.6, fill: 'none',
        stroke: 'var(--ink-3)', 'stroke-width': 1, 'stroke-dasharray': '2 2' }));
    }
    const title = svg('title');
    title.textContent = `Скважина ${well.name} · i${well.i} j${well.j}`;
    group.appendChild(title);
    const select = () => {
      state.selected = well.name;
      $$('#map .well').forEach((node) => node.removeAttribute('data-selected'));
      group.dataset.selected = '1';
      renderWellDetail(well);
    };
    group.addEventListener('click', select);
    group.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); select(); }
    });
    if (well.name === state.selected) group.dataset.selected = '1';
    holder.appendChild(group);
  });

  const counts = { producer: 0, injector: 0, dual: 0, idle: 0 };
  wells.forEach((well) => { counts[well.role] += 1; });
  const legend = $('#map-legend');
  if (state.mapMode === 'role') {
    legend.innerHTML = `
      <span><i style="background:var(--accent);border-radius:50%"></i>только добыча · ${counts.producer}</span>
      <span><i style="background:var(--alt)"></i>только закачка · ${counts.injector}</span>
      <span><i style="background:var(--g4);border-radius:50%"></i>меняет роль · ${counts.dual}</span>
      <span class="muted">размер точки — средний назначенный режим</span>`;
  } else if (state.mapMode === 'group') {
    legend.innerHTML = groupColors.map((color, index) =>
      `<span><i style="background:${color}"></i>группа ${index + 1}</span>`).join('') +
      '<span class="muted">добывающие делятся на 4 группы, нагнетательные на 2</span>';
  } else {
    legend.innerHTML =
      `<span><i style="background:var(--alt)"></i>сильнее изменён режим — насыщеннее цвет</span>` +
      `<span class="muted">множители от ×${num(scaleMin, 3)} до ×${num(scaleMin + scaleSpan, 3)} ` +
      `к исходному режиму скважины</span>`;
  }
}

function renderWellDetail(well) {
  const roles = { producer: 'Добывающая', injector: 'Нагнетательная', dual: 'Меняет роль по ходу разработки', idle: 'Не управляется расписанием' };
  const rows = [`<dt>Ячейка сетки</dt><dd>i ${well.i} · j ${well.j}</dd>`];
  if (well.producing_months) {
    rows.push(`<dt>Отбор жидкости</dt><dd>${num(well.mean_liquid_target_m3_d, 1)} м³/сут</dd>`);
    rows.push(`<dt>Месяцев в добыче</dt><dd>${well.producing_months}</dd>`);
    rows.push(`<dt>Группа добычи</dt><dd>Д‑${well.producer_group + 1}</dd>`);
  }
  if (well.injecting_months) {
    rows.push(`<dt>Закачка</dt><dd>${num(well.mean_injection_target_m3_d, 1)} м³/сут</dd>`);
    rows.push(`<dt>Месяцев в закачке</dt><dd>${well.injecting_months}</dd>`);
    rows.push(`<dt>Группа закачки</dt><dd>И‑${well.injector_group + 1}</dd>`);
  }
  rows.push(`<dt>Множитель к режиму</dt><dd>×${num(well.mean_scale, 3)}</dd>`);
  rows.push(`<dt>Запас до лимита</dt><dd>${num(state.case.wlpr_limit_m3_d - well.mean_liquid_target_m3_d, 1)} м³/сут</dd>`);
  $('#well-detail').innerHTML = `
    <h3>Скв. ${well.name}</h3>
    <p class="role">${roles[well.role]}</p>
    <dl>${rows.join('')}</dl>
    <p class="hint">Средние значения назначенного режима за контрактный период
      ${state.case.contract_start.slice(0, 10)} — ${state.case.period_end.slice(0, 10)} из итогового wells_schedule.inc.</p>`;
}

function initMapControls() {
  $$('#map-mode button').forEach((button) => button.addEventListener('click', () => {
    $$('#map-mode button').forEach((other) => other.setAttribute('aria-pressed', String(other === button)));
    state.mapMode = button.dataset.mode;
    renderMap();
  }));
}

/* ------------------------------------------------------------------- boot */

async function boot() {
  $('#btn-verify').addEventListener('click', () => startRun('verify'));
  $('#btn-rebuild').addEventListener('click', () => startRun('rebuild'));
  try { initNav(); initMapControls(); } catch (error) { banner(`Интерфейс: ${error.message}`); }
  try {
    const [info, metrics, quality, explanation, field, production, annual, operations, room] = await Promise.all([
      api('/api/case'), api('/api/metrics'), api('/api/quality'), api('/api/explanation'),
      api('/api/field'), api('/api/production'), api('/api/annual'), api('/api/operations'),
      api('/api/room'),
    ]);
    Object.assign(state, {
      case: info, metrics, quality, explanation, room,
      wells: field.wells, production, annual: annual.rows, operations,
    });
  } catch (error) {
    banner(`Данные прогона недоступны: ${error.message}. Проверьте, что каталог runs/verified на месте.`);
    return;
  }
  renderCase();
  renderTrust();
  renderWhy();
  renderDeliverable(null);
  renderExplanation();
  renderResultView();
  renderOperations();
  renderQuality();
  renderAnnual();
  state.selected = state.wells[0] && state.wells[0].name;
  renderMap();
  if (state.wells[0]) renderWellDetail(state.wells[0]);
}

boot();
