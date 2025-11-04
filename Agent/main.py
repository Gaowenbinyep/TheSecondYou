from graph_nodes.nodes import build_workflow


if __name__ == "__main__":
    workflow = build_workflow()
    result = workflow.invoke({})
