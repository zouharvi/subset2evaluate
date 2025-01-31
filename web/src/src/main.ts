import { DEVMODE } from "./globals"
import ITEMS from "./data.json"

let selected = new Set<number>()
const BUDGET = 10
let revealed = false

$("#button_reveal").on("click", function () {
  revealed = !revealed
  $("#div_results_total").css("visibility", revealed ? "visible" : "hidden")
  $("#span_correlation").css("visibility", revealed ? "visible" : "hidden")
  if (revealed) {
    // rho symbol
    $("#button_reveal").val(`👆 Hide evaluation on all items`)
  } else {
    $("#button_reveal").val("👆 Reveal evaluation on all items")
  }
})
$("#div_results_total").css("visibility", "hidden")
$("#span_correlation").css("visibility", "hidden")

let models = Object.keys(ITEMS[0]["scores"])
let model_to_emoji = new Map<string, string>(
  models.map((s, i) => [s, `img/robot${i + 1}.svg`])
)
let model_to_name = {
  "ONLINE-W": "Public-A",
  "IOL_Research": "IOL Research",
  "Unbabel-Tower70B": "Tower70b",
  "IKUN-C": "IKUN-C",
  "CUNI-GA": "CUNI-GA",
}
// total model average
let model_avg_total: { [model: string]: number } = {}
for (const model of models) {
  let scores = ITEMS.map((item) => item["scores"][model])
  let avg = scores.reduce((a, b) => a + b, 0) / scores.length
  model_avg_total[model] = avg
}
let out = "<div style='text-align: center; width: 100%; font-weight: bold;'>Based on all items</div><br><table>"
out += "<tr><th>Model</th><th>Score</th></tr>"
let sorted = Object.entries(model_avg_total).sort((a, b) => b[1] - a[1])
for (const [model, score] of sorted) {
  out += `<tr><td><img src="${model_to_emoji.get(model)}" style="width: 20px; height: 20px; margin-right: 2px;">${model_to_name[model]}</td>`
  out += `<td>${(score * 100).toFixed(1)}%</td></tr>`
}
out += "</table>"
$("#div_results_total").html(out)

function recompute_table() {
  if (selected.size == 0) {
    // $("#div_results").html("First select a few items to evaluate.")
    $("#div_results").html("")
    $("#span_correlation").css("visibility", "hidden")
    return
  }
  if (revealed) {
      $("#span_correlation").css("visibility", "visible")
  }

  // redraw table
  let out = "<div style='text-align: center; width: 100%; font-weight: bold;'>Based on selection</div><br><table>"
  // compute average per each model
  let model_scores: { [model: string]: Array<number> } = {}
  for (const i of selected) {
    let item = ITEMS[i]
    for (const model in item["scores"]) {
      if (model_scores[model] === undefined) {
        model_scores[model] = []
      }
      model_scores[model].push(item["scores"][model])
    }
  }
  // compute average and sort
  let model_avg: { [model: string]: number } = {}
  for (const model in model_scores) {
    let scores = model_scores[model]
    let avg = scores.reduce((a, b) => a + b, 0) / scores.length
    model_avg[model] = avg
  }
  let sorted = Object.entries(model_avg).sort((a, b) => b[1] - a[1])
  // print table
  out += "<tr><th>Model</th><th>Score</th></tr>"
  for (const [model, score] of sorted) {
    out += `<tr><td><img src="${model_to_emoji.get(model)}" style="width: 20px; height: 20px; margin-right: 2px;">${model_to_name[model]}</td>`
    out += `<td>${(score * 100).toFixed(1)}%</td></tr>`
  }

  out += "</table>"
  $("#div_results").html(out)

  // compute rank correlation between model_avg and model_avg_total
  let x = []
  let y = []
  for (const model in model_avg_total) {
    x.push(model_avg_total[model])
    y.push(model_avg[model])
  }
  $("#span_correlation").text(`Correlation between rankings is ${(spearman(x, y)*100).toFixed(1)}%`)
}

function prepare_item_div(
  item: { "src": string, "scores": { [model: string]: number } },
  item_i: number
): JQuery<HTMLElement> {
  let el = $('<div class="item-div">')
  el.text(item["src"])
  // toggle class on click
  el.on("click", function () {
    if (selected.size == BUDGET && !selected.has(item_i))
      return

    // update tracking of items
    if (selected.has(item_i)) {
      selected.delete(item_i)
    } else {
      selected.add(item_i)
    }

    // update budget indicator
    let used = selected.size
    if (used == BUDGET) {
      $("#span_budget").text(`no items`)
    } else if (used == BUDGET - 1) {
      $("#span_budget").text(`1 item`)
    } else {
      $("#span_budget").text(`${BUDGET - used} items`)
    }

    recompute_table()

    el.toggleClass("item-div-selected")
  })
  return el
}

let el_items = $("#div_items")
for (const item_i in ITEMS) {
  el_items.append(prepare_item_div(ITEMS[item_i], parseInt(item_i)))
}

function spearman(x: any[], y: any[]) {
  let n = x.length
  let x_rank = x.map((_, i) => i).sort((a, b) => x[a] - x[b])
  let y_rank = y.map((_, i) => i).sort((a, b) => y[a] - y[b])
  let d2 = 0
  for (let i = 0; i < n; i++) {
    d2 += (x_rank[i] - y_rank[i]) ** 2
  }
  let rho = 1 - 6 * d2 / (n * (n ** 2 - 1))
  return rho
}
