"""
Debug script to find the BaseModel initialization error
Run this file directly: python debug.py
"""
from __future__ import annotations
import json
import os
import traceback
from rich import print as rprint
from rich.panel import Panel
from rich.pretty import Pretty

# Import your modules
from graphs import graph, GraphState
from models import ResearchPlan, SourceSummary, FinalBrief

# ==========================
# Debug each node individually
# ==========================
def test_each_node_individually():
    """Test each node function separately to isolate the error"""
    rprint(Panel.fit("[bold yellow]Testing Each Node Individually[/bold yellow]"))
    
    # Create test state
    test_state = {
        "topic": "AI in Healthcare",
        "depth": 2,
        "audience": "general",
        "follow_up": False,
        "user_id": "test_user"
    }
    
    # Import node functions directly
    from graphs import context_node, planning_node, search_node, synthesis_node, post_node
    
    # Test each node
    nodes_to_test = [
        ("context", context_node),
        ("planning", planning_node), 
        ("search", search_node),
        ("synthesis", synthesis_node),
        ("post", post_node)
    ]
    
    working_state = test_state.copy()
    
    for node_name, node_func in nodes_to_test:
        try:
            rprint(f"\n[blue]Testing {node_name} node...[/blue]")
            result_state = node_func(working_state)
            rprint(f"[green]✓ {node_name} node works[/green]")
            
            # Update working state for next node
            working_state = result_state
            
            # Show what was added to state
            new_keys = set(result_state.keys()) - set(test_state.keys())
            if new_keys:
                rprint(f"  Added keys: {list(new_keys)}")
                
        except Exception as e:
            rprint(f"[red]✗ {node_name} node failed:[/red] {str(e)}")
            rprint("[yellow]Full traceback:[/yellow]")
            traceback.print_exc()
            return node_name, str(e)
    
    return None, None

def test_model_creation_directly():
    """Test creating each Pydantic model directly"""
    rprint(Panel.fit("[bold yellow]Testing Model Creation Directly[/bold yellow]"))
    
    try:
        # Test ResearchPlan
        rprint("[blue]Testing ResearchPlan...[/blue]")
        from models import make_mock_plan
        plan = make_mock_plan("Test Topic", 2)
        rprint("[green]✓ ResearchPlan creation works[/green]")
        
        # Test SourceSummary
        rprint("[blue]Testing SourceSummary...[/blue]")
        from models import make_mock_source_summaries
        summaries = make_mock_source_summaries("Test Topic", 2)
        rprint("[green]✓ SourceSummary creation works[/green]")
        
        # Test FinalBrief
        rprint("[blue]Testing FinalBrief...[/blue]")
        from models import compile_final_brief
        brief = compile_final_brief("Test Topic", 2, "general", summaries)
        rprint("[green]✓ FinalBrief creation works[/green]")
        
        return True
        
    except Exception as e:
        rprint(f"[red]✗ Model creation failed:[/red] {str(e)}")
        traceback.print_exc()
        return False

def debug_pydantic_models():
    """Debug Pydantic model initialization patterns"""
    rprint(Panel.fit("[bold yellow]Debugging Pydantic Model Patterns[/bold yellow]"))
    
    try:
        from models import ResearchPlanStep
        
        # Test different initialization patterns
        rprint("[blue]Testing ResearchPlanStep initialization...[/blue]")
        
        # Method 1: Keyword arguments (correct)
        try:
            step1 = ResearchPlanStep(
                order=1,
                objective="test",
                method="test",
                expected_output="test"
            )
            rprint("[green]✓ Keyword arguments work[/green]")
        except Exception as e:
            rprint(f"[red]✗ Keyword arguments failed:[/red] {e}")
        
        # Method 2: Positional arguments (this might be the issue)
        try:
            step2 = ResearchPlanStep(1, "test", "test", "test")
            rprint("[yellow]⚠ Positional arguments work (unexpected!)[/yellow]")
        except Exception as e:
            rprint(f"[red]✗ Positional arguments failed (expected):[/red] {e}")
            
    except Exception as e:
        rprint(f"[red]Error in debug_pydantic_models:[/red] {e}")
        traceback.print_exc()

def inspect_source_code():
    """Look at the actual source code for problematic patterns"""
    rprint(Panel.fit("[bold yellow]Inspecting Source Code[/bold yellow]"))
    
    import inspect
    from models import make_mock_plan, make_mock_source_summaries, compile_final_brief
    
    functions_to_check = [
        ("make_mock_plan", make_mock_plan),
        ("make_mock_source_summaries", make_mock_source_summaries), 
        ("compile_final_brief", compile_final_brief)
    ]
    
    for func_name, func in functions_to_check:
        try:
            rprint(f"\n[blue]Checking {func_name}:[/blue]")
            source = inspect.getsource(func)
            
            # Look for model instantiation lines
            lines = source.split('\n')
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                if any(model in stripped for model in ['ResearchPlanStep(', 'SourceSummary(', 'FinalBrief(', 'FinalBriefSection(', 'FinalBriefReference(']):
                    rprint(f"  Line {i}: {stripped}")
                    
        except Exception as e:
            rprint(f"[red]Could not inspect {func_name}:[/red] {e}")

def full_debug_run():
    """Run comprehensive debug"""
    rprint(Panel.fit("[bold cyan]🔍 COMPREHENSIVE DEBUG SESSION 🔍[/bold cyan]"))
    
    # Step 1: Test model creation
    rprint("\n" + "="*50)
    rprint("[bold]STEP 1: Test Model Creation[/bold]")
    models_ok = test_model_creation_directly()
    
    if not models_ok:
        rprint("[red]❌ Model creation failed - stopping here[/red]")
        return
    
    # Step 2: Debug Pydantic patterns
    rprint("\n" + "="*50)
    rprint("[bold]STEP 2: Debug Pydantic Patterns[/bold]")
    debug_pydantic_models()
    
    # Step 3: Inspect source code
    rprint("\n" + "="*50)
    rprint("[bold]STEP 3: Inspect Source Code[/bold]")
    inspect_source_code()
    
    # Step 4: Test nodes individually
    rprint("\n" + "="*50)
    rprint("[bold]STEP 4: Test Each Node Individually[/bold]")
    failed_node, error = test_each_node_individually()
    
    if failed_node:
        rprint(f"\n[red]🎯 FOUND THE ISSUE![/red]")
        rprint(f"[red]Failed node: {failed_node}[/red]")
        rprint(f"[red]Error: {error}[/red]")
        rprint("[yellow]Check the source code of that node function![/yellow]")
    else:
        rprint(f"\n[green]🤔 All individual nodes work...[/green]")
        rprint("[yellow]The issue might be in graph execution or state passing[/yellow]")
        
        # Try full graph execution
        rprint("\n[blue]Testing full graph execution...[/blue]")
        try:
            test_state = {
                "topic": "AI in Healthcare",
                "depth": 2,
                "audience": "general",
                "follow_up": False,
                "user_id": "test_user"
            }
            compiled_graph = graph.compile()
            final_state = compiled_graph.invoke(test_state)
            rprint("[green]✓ Full graph execution works![/green]")
        except Exception as e:
            rprint(f"[red]✗ Full graph execution failed:[/red] {e}")
            traceback.print_exc()

if __name__ == "__main__":
    full_debug_run()