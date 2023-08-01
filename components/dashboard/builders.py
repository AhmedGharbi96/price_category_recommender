from dash import dash_table, html

from components.dashboard.config import DashboardConfig
from components.dashboard.ui_components.local_explainer import LocalExplainerComponent


def build_local_explainer_component(config: DashboardConfig):
    local_explainer_component = LocalExplainerComponent(config)
    chart = local_explainer_component.build()
    local_explainer_component.register_callbacks()
    return chart


def build_table(df, id, row_selectable=False):
    return html.Div(
        style={
            "display": "flex",
            "justify-content": "center",
            "align-items": "center",
        },
        children=dash_table.DataTable(
            df.to_dict("records"),
            [{"name": i, "id": i} for i in df.columns],
            row_selectable=row_selectable,
            style_table={
                "overflowX": "scroll",
                "max-height": "500px",
                "max-width": "1800px",
            },
            style_header={
                "color": "white",
                "backgroundColor": "grey",
                "fontWeight": "bold",
            },
            style_cell={
                "textAlign": "center",
                "whiteSpace": "normal",
                "height": "auto",
                "width": "200px",
                "backgroundColor": "HoneyDew",
            },
            id=id,
        ),
    )
