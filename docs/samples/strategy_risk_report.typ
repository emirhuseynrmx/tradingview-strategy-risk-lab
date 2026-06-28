#set page(margin: 15mm)
#set text(font: "Arial", size: 10pt)
#set heading(numbering: none)

#let ink = rgb("#111827")
#let muted = rgb("#64748b")
#let panel = rgb("#f3f6fb")
#let line = rgb("#d7dee8")
#let risk = rgb("#b91c1c")

#let card(label, value, note: none) = block[
  #rect(fill: panel, radius: 6pt, inset: 9pt, width: 100%)[
    #text(size: 8pt, fill: muted, weight: "bold")[#upper(label)]
    #linebreak()
    #text(size: 19pt, fill: ink, weight: "bold")[#value]
    #if note != none [#linebreak() #text(size: 8pt, fill: muted)[#note]]
  ]
]

#let check(label, value) = block[
  #text(weight: "bold")[#label]
  #linebreak()
  #text(fill: muted)[#value]
]

#text(size: 22pt, weight: "bold", fill: ink)[TradingView Strategy Risk Lab]

#text(fill: muted)[Model-training report for a TradingView trade export. The report reviews fragile setups with holdout metrics, SHAP drivers, and a zero-lookahead feature boundary.]

#grid(columns: (1fr, 1fr, 1fr), gutter: 8pt)[
  #card("Trades", "240", note: "validated rows")
][
  #card("Bad-trade rate", "0.4833", note: "label prevalence")
][
  #card("ROC-AUC", "0.5873", note: "holdout score")
]

#v(10pt)
#grid(columns: (1.45fr, 1fr), gutter: 14pt)[
  #text(size: 13pt, weight: "bold")[Top Risk Drivers]
  #v(4pt)
  #table(
    columns: (1fr, .7fr),
    stroke: line,
    inset: 5pt,
    [*Feature*], [*Importance*],
  [trend_score], [0.38024],
  [atr_pct], [0.34524],
  [volume_z], [0.23011],
  [side_code], [0.04441],
  )
][
  #text(size: 13pt, weight: "bold")[Evidence Contract]
  #v(4pt)
  #block(stroke: line, radius: 6pt, inset: 8pt)[
    #check("Feature timing", "Only setup fields known before exit are model inputs.")
    #v(5pt)
    #check("Label boundary", "Realized PnL is used only to define the target.")
    #v(5pt)
    #check("Explainability", "SHAP shows drivers; DiCE stays on actionable fields.")
    #v(5pt)
    #check("Usage", "This is strategy review, not live execution.")
  ]
]

#v(10pt)
#block(fill: rgb("#fef2f2"), radius: 6pt, inset: 8pt)[
  #text(weight: "bold", fill: risk)[Counterfactual next step:] Start the counterfactual review with trend_score. Compare the bad-trade rate before and after tightening that condition before using it as a live filter.
]
