# visualize_graph.py
from graphs import graph

# Step 1: Compile the graph
app = graph.compile()

# Step 2: Get the Graph object
graph_obj = app.get_graph()

# Step 3: Export as Mermaid PNG
png_data = graph_obj.draw_mermaid_png()

# Step 4: Save to file
with open("research_graph_structure.png", "wb") as f:
    f.write(png_data)

print("Graph visualization saved as research_graph_structure.png")
