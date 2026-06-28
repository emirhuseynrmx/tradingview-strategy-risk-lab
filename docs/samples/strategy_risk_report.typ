#set page(margin: 16mm)
#set text(font: "Arial", size: 10pt)

#text(size: 18pt, weight: "bold")[TradingView Strategy Risk Lab]

Model-training report for a TradingView trade export.

#grid(columns: (1fr, 1fr, 1fr), gutter: 8pt)[
  #block(fill: rgb("#f3f6fb"), radius: 4pt, inset: 8pt)[Trades\ #text(size: 18pt, weight: "bold")[240]]
][
  #block(fill: rgb("#f3f6fb"), radius: 4pt, inset: 8pt)[Bad-trade rate\ #text(size: 18pt, weight: "bold")[0.4833]]
][
  #block(fill: rgb("#f3f6fb"), radius: 4pt, inset: 8pt)[ROC-AUC\ #text(size: 18pt, weight: "bold")[0.5873]]
]

#v(10pt)
#text(size: 12pt, weight: "bold")[Top Risk Drivers]

#table(columns: (1fr, 1fr), [Feature], [Importance],
  [trend_score], [0.38024],
  [atr_pct], [0.34524],
  [volume_z], [0.23011],
  [side_code], [0.04441],
)

#v(8pt)
#text(weight: "bold")[Counterfactual next step:] Start the counterfactual review with trend_score. Compare the bad-trade rate before and after tightening that condition before using it as a live filter.
