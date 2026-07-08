(function () {
  const data = window.KLINE_DATA;
  const fields = data.fields;
  const index = Object.fromEntries(fields.map((field, i) => [field, i]));

  const state = {
    code: data.codes[0],
    range: 120,
    pointerIndex: null,
  };

  const canvas = document.getElementById("klineCanvas");
  const tooltip = document.getElementById("tooltip");
  const codeSelect = document.getElementById("codeSelect");
  const codeSearch = document.getElementById("codeSearch");
  const recentRows = document.getElementById("recentRows");
  const ctx = canvas.getContext("2d");

  const formatNumber = new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 2 });
  const formatInteger = new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 0 });

  function rowValue(row, field) {
    return row[index[field]];
  }

  function formatAmount(value) {
    if (!Number.isFinite(value)) return "--";
    if (Math.abs(value) >= 100000000) return `${formatNumber.format(value / 100000000)} 亿`;
    if (Math.abs(value) >= 10000) return `${formatNumber.format(value / 10000)} 万`;
    return formatNumber.format(value);
  }

  function formatPrice(value) {
    return Number.isFinite(value) ? formatNumber.format(value) : "--";
  }

  function getSeries() {
    const rows = data.series[state.code] || [];
    const start = state.range === "all" ? 0 : Math.max(0, rows.length - Number(state.range));
    return rows.slice(start).map((row, offset) => ({
      date: data.dates[start + offset],
      row,
      open: rowValue(row, "open"),
      high: rowValue(row, "high"),
      low: rowValue(row, "low"),
      close: rowValue(row, "close"),
      volume: rowValue(row, "volume"),
      amount: rowValue(row, "amount"),
      trade_count: rowValue(row, "trade_count"),
      buy_volume: rowValue(row, "buy_volume"),
      sell_volume: rowValue(row, "sell_volume"),
      buy_amount: rowValue(row, "buy_amount"),
      sell_amount: rowValue(row, "sell_amount"),
    }));
  }

  function populateCodes(codes) {
    codeSelect.replaceChildren();
    for (const code of codes) {
      const option = document.createElement("option");
      option.value = code;
      option.textContent = code;
      codeSelect.appendChild(option);
    }
    codeSelect.value = state.code;
  }

  function setActiveRange() {
    document.querySelectorAll("[data-range]").forEach((button) => {
      button.classList.toggle("active", String(state.range) === button.dataset.range);
    });
  }

  function updateSummary(series) {
    const last = series[series.length - 1];
    const prev = series[series.length - 2];
    const ret = last && prev ? last.close / prev.close - 1 : NaN;

    document.getElementById("statCode").textContent = state.code;
    document.getElementById("statDate").textContent = last ? last.date : "--";
    document.getElementById("statClose").textContent = last ? formatPrice(last.close) : "--";
    document.getElementById("statReturn").textContent = Number.isFinite(ret)
      ? `${ret >= 0 ? "+" : ""}${(ret * 100).toFixed(2)}%`
      : "--";
    document.getElementById("statReturn").className = ret >= 0 ? "up" : "down";
    document.getElementById("statVolume").textContent = last ? formatInteger.format(last.volume) : "--";
    document.getElementById("statAmount").textContent = last ? formatAmount(last.amount) : "--";
    document.getElementById("statBuyVolume").textContent = last ? formatInteger.format(last.buy_volume) : "--";
    document.getElementById("statSellVolume").textContent = last ? formatInteger.format(last.sell_volume) : "--";
    document.getElementById("statBuyAmount").textContent = last ? formatAmount(last.buy_amount) : "--";
    document.getElementById("statSellAmount").textContent = last ? formatAmount(last.sell_amount) : "--";
  }

  function updateTable(series) {
    recentRows.replaceChildren();
    const rows = series.slice(-12).reverse();
    for (const item of rows) {
      const tr = document.createElement("tr");
      const up = item.close >= item.open;
      tr.innerHTML = `
        <td>${item.date}</td>
        <td>${formatPrice(item.open)}</td>
        <td>${formatPrice(item.high)}</td>
        <td>${formatPrice(item.low)}</td>
        <td class="${up ? "up" : "down"}">${formatPrice(item.close)}</td>
        <td>${formatInteger.format(item.volume)}</td>
        <td>${formatInteger.format(item.trade_count)}</td>
        <td>${formatAmount(item.amount)}</td>
        <td>${formatInteger.format(item.buy_volume)}</td>
        <td>${formatInteger.format(item.sell_volume)}</td>
        <td>${formatAmount(item.buy_amount)}</td>
        <td>${formatAmount(item.sell_amount)}</td>
      `;
      recentRows.appendChild(tr);
    }
  }

  function setCanvasSize() {
    const ratio = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = Math.max(1, Math.floor(rect.width * ratio));
    canvas.height = Math.max(1, Math.floor(rect.height * ratio));
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  }

  function drawGrid(area, min, max) {
    ctx.strokeStyle = "#edf0f4";
    ctx.fillStyle = "#6b7280";
    ctx.lineWidth = 1;
    ctx.font = "12px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
    for (let i = 0; i <= 4; i += 1) {
      const y = area.top + (area.height * i) / 4;
      const value = max - ((max - min) * i) / 4;
      ctx.beginPath();
      ctx.moveTo(area.left, y);
      ctx.lineTo(area.right, y);
      ctx.stroke();
      ctx.fillText(formatPrice(value), area.right + 8, y + 4);
    }
  }

  function render() {
    const series = getSeries();
    setCanvasSize();

    const rect = canvas.getBoundingClientRect();
    ctx.clearRect(0, 0, rect.width, rect.height);

    if (series.length === 0) return;

    updateSummary(series);
    updateTable(series);

    const padding = { top: 18, right: 74, bottom: 28, left: 14 };
    const priceArea = {
      left: padding.left,
      top: padding.top,
      right: rect.width - padding.right,
      height: Math.round((rect.height - padding.top - padding.bottom) * 0.72),
    };
    priceArea.bottom = priceArea.top + priceArea.height;
    priceArea.width = priceArea.right - priceArea.left;

    const volumeArea = {
      left: padding.left,
      top: priceArea.bottom + 22,
      right: priceArea.right,
      bottom: rect.height - padding.bottom,
    };
    volumeArea.width = volumeArea.right - volumeArea.left;
    volumeArea.height = volumeArea.bottom - volumeArea.top;

    const highs = series.map((item) => item.high).filter(Number.isFinite);
    const lows = series.map((item) => item.low).filter(Number.isFinite);
    const maxPrice = Math.max(...highs);
    const minPrice = Math.min(...lows);
    const pricePad = Math.max((maxPrice - minPrice) * 0.08, 0.01);
    const priceMax = maxPrice + pricePad;
    const priceMin = Math.max(0, minPrice - pricePad);
    const maxVolume = Math.max(...series.map((item) => item.volume || 0), 1);
    const step = priceArea.width / series.length;
    const candleWidth = Math.max(2, Math.min(12, step * 0.62));

    const yPrice = (value) =>
      priceArea.top + ((priceMax - value) / (priceMax - priceMin)) * priceArea.height;
    const yVolume = (value) => volumeArea.bottom - (value / maxVolume) * volumeArea.height;
    const xAt = (i) => priceArea.left + step * i + step / 2;

    drawGrid(priceArea, priceMin, priceMax);

    ctx.strokeStyle = "#d8dde6";
    ctx.beginPath();
    ctx.moveTo(volumeArea.left, volumeArea.top);
    ctx.lineTo(volumeArea.right, volumeArea.top);
    ctx.stroke();

    series.forEach((item, i) => {
      const x = xAt(i);
      const up = item.close >= item.open;
      const color = up ? "#d92d20" : "#079455";
      const openY = yPrice(item.open);
      const closeY = yPrice(item.close);
      const highY = yPrice(item.high);
      const lowY = yPrice(item.low);
      const bodyTop = Math.min(openY, closeY);
      const bodyHeight = Math.max(1, Math.abs(closeY - openY));

      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.moveTo(x, highY);
      ctx.lineTo(x, lowY);
      ctx.stroke();
      ctx.fillRect(x - candleWidth / 2, bodyTop, candleWidth, bodyHeight);

      const volumeTop = yVolume(item.volume || 0);
      ctx.globalAlpha = 0.45;
      ctx.fillRect(x - candleWidth / 2, volumeTop, candleWidth, volumeArea.bottom - volumeTop);
      ctx.globalAlpha = 1;
    });

    ctx.fillStyle = "#6b7280";
    ctx.font = "12px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
    ctx.fillText(formatAmount(maxVolume), volumeArea.right + 8, volumeArea.top + 4);
    ctx.fillText("0", volumeArea.right + 8, volumeArea.bottom);

    const labelEvery = Math.max(1, Math.ceil(series.length / 6));
    ctx.fillStyle = "#6b7280";
    series.forEach((item, i) => {
      if (i % labelEvery !== 0 && i !== series.length - 1) return;
      const x = xAt(i);
      ctx.fillText(item.date.slice(4), Math.max(priceArea.left, x - 18), rect.height - 8);
    });

    if (state.pointerIndex !== null && state.pointerIndex >= 0 && state.pointerIndex < series.length) {
      const item = series[state.pointerIndex];
      const x = xAt(state.pointerIndex);
      ctx.strokeStyle = "#94a3b8";
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(x, priceArea.top);
      ctx.lineTo(x, volumeArea.bottom);
      ctx.stroke();
      ctx.setLineDash([]);
      tooltip.hidden = false;
      tooltip.innerHTML = `
        <strong>${state.code} / ${item.date}</strong><br />
        开 ${formatPrice(item.open)}　高 ${formatPrice(item.high)}<br />
        低 ${formatPrice(item.low)}　收 ${formatPrice(item.close)}<br />
        量 ${formatInteger.format(item.volume)}　笔 ${formatInteger.format(item.trade_count)}<br />
        额 ${formatAmount(item.amount)}<br />
        主买量 ${formatInteger.format(item.buy_volume)}　主卖量 ${formatInteger.format(item.sell_volume)}<br />
        主买额 ${formatAmount(item.buy_amount)}　主卖额 ${formatAmount(item.sell_amount)}
      `;
      const tooltipLeft = Math.min(rect.width - 200, Math.max(8, x + 12));
      tooltip.style.left = `${tooltipLeft}px`;
      tooltip.style.top = `${priceArea.top + 12}px`;
    } else {
      tooltip.hidden = true;
    }
  }

  function handlePointer(event) {
    const series = getSeries();
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const left = 14;
    const right = rect.width - 74;
    const step = (right - left) / series.length;
    state.pointerIndex = Math.max(0, Math.min(series.length - 1, Math.floor((x - left) / step)));
    render();
  }

  function init() {
    populateCodes(data.codes);
    document.getElementById("dataRange").textContent =
      `${data.dates[0]} - ${data.dates[data.dates.length - 1]} / ${data.codes.length} 只股票`;

    codeSelect.addEventListener("change", () => {
      state.code = codeSelect.value;
      state.pointerIndex = null;
      render();
    });

    codeSearch.addEventListener("input", () => {
      const query = codeSearch.value.trim();
      const codes = query ? data.codes.filter((code) => code.includes(query)).slice(0, 80) : data.codes;
      populateCodes(codes.length ? codes : data.codes);
    });

    document.querySelectorAll("[data-range]").forEach((button) => {
      button.addEventListener("click", () => {
        state.range = button.dataset.range === "all" ? "all" : Number(button.dataset.range);
        state.pointerIndex = null;
        setActiveRange();
        render();
      });
    });

    canvas.addEventListener("mousemove", handlePointer);
    canvas.addEventListener("mouseleave", () => {
      state.pointerIndex = null;
      render();
    });

    new ResizeObserver(render).observe(canvas);
    setActiveRange();
    render();
  }

  window.addEventListener("DOMContentLoaded", init);
})();
