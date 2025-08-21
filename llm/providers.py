# llm/providers.py
"""
LLM provider integration that works with your existing models.py schemas.
"""

import os
import logging
from typing import Dict, Any, Optional, List
from enum import Enum

# LangChain imports
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable

try:
    from langchain.output_parsers import OutputFixingParser
except ImportError:
    OutputFixingParser = None

# Import your existing models
from models import ResearchPlan, SourceSummary, FinalBrief

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

try:
    from langchain_mistralai import ChatMistralAI
except ImportError:
    ChatMistralAI = None

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    GROK = "grok"
    MISTRAL = "mistral"


class LLMManager:
    """Manages LLM providers for the research brief system."""
    
    def __init__(self):
        self.providers: Dict[LLMProvider, BaseLanguageModel] = {}
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize available LLM providers."""
        
        # Grok (via xAI API - compatible with OpenAI client)
        if os.getenv("XAI_API_KEY") and ChatOpenAI:
            try:
                llm = ChatOpenAI(
                    model=os.getenv("GROK_MODEL", "grok-beta"),
                    openai_api_base="https://api.x.ai/v1",
                    openai_api_key=os.getenv("XAI_API_KEY"),
                    temperature=0.1,
                    max_tokens=2000
                )
                self.providers[LLMProvider.GROK] = llm
                logger.info("Initialized Grok provider via xAI API")
            except Exception as e:
                logger.warning(f"Failed to initialize Grok: {e}")
        
        # Mistral (via official Mistral API)
        if os.getenv("MISTRAL_API_KEY") and ChatMistralAI:
            try:
                llm = ChatMistralAI(
                    model=os.getenv("MISTRAL_MODEL", "mistral-small-latest"),
                    mistral_api_key=os.getenv("MISTRAL_API_KEY"),
                    temperature=0.1,
                    max_tokens=2000,
                    # Additional Mistral parameters
                    safe_prompt=False,  # Allow more flexible prompting
                    random_seed=42,     # For reproducible results
                )
                self.providers[LLMProvider.MISTRAL] = llm
                logger.info("Initialized Mistral provider via official API")
            except Exception as e:
                logger.warning(f"Failed to initialize Mistral: {e}")
        
        if not self.providers:
            logger.warning("No LLM providers available. Using mock responses.")
            logger.info("Available providers: Grok (XAI_API_KEY), Mistral (MISTRAL_API_KEY)")
    
    def get_primary_llm(self) -> Optional[BaseLanguageModel]:
        """Get the primary LLM (prefer Grok, then Mistral)."""
        # Priority order for API models
        priority_order = [
            LLMProvider.GROK,
            LLMProvider.MISTRAL
        ]
        
        for provider in priority_order:
            if provider in self.providers:
                return self.providers[provider]
        return None
    
    def has_llm(self) -> bool:
        """Check if any LLM is available."""
        return len(self.providers) > 0
    
    def get_available_providers(self) -> List[str]:
        """Get list of available provider names."""
        return [provider.value for provider in self.providers.keys()]


# Global LLM manager
_llm_manager = None

def get_llm_manager() -> LLMManager:
    """Get the global LLM manager instance."""
    global _llm_manager
    if _llm_manager is None:
        _llm_manager = LLMManager()
    return _llm_manager


def create_research_plan_with_llm(topic: str, depth: int) -> ResearchPlan:
    """Generate a research plan using LLM or fallback to mock."""
    manager = get_llm_manager()
    llm = manager.get_primary_llm()
    
    if not llm:
        logger.warning("No LLM available, using mock research plan")
        from models import make_mock_plan
        return make_mock_plan(topic, depth)
    
    try:
        # Create structured output parser
        parser = PydanticOutputParser(pydantic_object=ResearchPlan)
        
        # Enhanced prompt template for better results with free models
        prompt = PromptTemplate(
            template="""You are a research planning expert. Create a detailed research plan for the given topic.

Topic: {topic}
Depth Level: {depth}/5 (1=basic overview, 2=moderate detail, 3=detailed analysis, 4=comprehensive study, 5=expert-level research)

Create a structured research plan with these components:
- topic: "{topic}"
- depth: {depth}
- steps: A list of research steps in logical order

Each step should have:
- order: Sequential number (1, 2, 3, ...)
- objective: Clear goal for this step
- method: Research method ("search", "analysis", "synthesis", "validation")
- expected_output: What this step should produce

Guidelines:
- Depth 1-2: 3-5 basic steps focusing on overview
- Depth 3: 5-7 steps with moderate analysis  
- Depth 4-5: 7-10+ steps with deep analysis and validation

Return ONLY the JSON structure requested below.

{format_instructions}
""",
            input_variables=["topic", "depth"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        # Create and run chain with error handling for different model types
        try:
            chain = prompt | llm | parser
            result = chain.invoke({"topic": topic, "depth": depth})
        except Exception as parse_error:
            # Fallback for models that don't handle structured output well
            logger.warning(f"Structured parsing failed: {parse_error}")
            response = llm.invoke(prompt.format(topic=topic, depth=depth, format_instructions=""))
            
            # Manual parsing fallback
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
            
            # Try to extract JSON or create structured response
            result = _parse_research_plan_response(response_text, topic, depth)
        
        logger.info(f"Generated research plan with {len(result.steps)} steps")
        return result
        
    except Exception as e:
        logger.error(f"Failed to generate research plan with LLM: {e}")
        logger.info("Falling back to mock research plan")
        from models import make_mock_plan
        return make_mock_plan(topic, depth)


def _parse_research_plan_response(response_text: str, topic: str, depth: int) -> ResearchPlan:
    """Parse LLM response into ResearchPlan structure."""
    import json
    import re
    from models import ResearchStep
    
    # Try to find JSON in response
    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            if 'steps' in data and isinstance(data['steps'], list):
                steps = []
                for i, step_data in enumerate(data['steps'], 1):
                    if isinstance(step_data, dict):
                        steps.append(ResearchStep(
                            order=step_data.get('order', i),
                            objective=step_data.get('objective', f"Research step {i}"),
                            method=step_data.get('method', 'search'),
                            expected_output=step_data.get('expected_output', f"Output for step {i}")
                        ))
                
                if steps:
                    return ResearchPlan(topic=topic, depth=depth, steps=steps)
        except json.JSONDecodeError:
            pass
    
    # Fallback to mock if parsing fails
    from models import make_mock_plan
    return make_mock_plan(topic, depth)


def create_source_summary_with_llm(
    source_id: str, 
    url: str, 
    title: str, 
    content: str, 
    topic: str
) -> SourceSummary:
    """Create a source summary using LLM or fallback to mock."""
    manager = get_llm_manager()
    llm = manager.get_primary_llm()
    
    if not llm or not content:
        logger.warning("No LLM available or no content, creating basic source summary")
        from datetime import datetime
        return SourceSummary(
            source_id=source_id,
            url=url,
            title=title or "Untitled Source",
            key_points=[f"Key information about {topic} from this source"],
            credibility_notes="Generated without full content analysis",
            extracted_at=datetime.utcnow()
        )
    
    try:
        # Truncate content if too long (important for free models)
        max_content_length = 2000  # Reduced for free models
        if len(content) > max_content_length:
            content = content[:max_content_length] + "... [truncated]"
        
        # Simplified prompt for better compatibility with free models
        prompt_template = """Analyze this source for research on: {topic}

Source: {title}
URL: {url}
Content: {content}

Extract 3-5 key points relevant to "{topic}".
Assess the source's credibility and reliability.

Format your response as:
KEY POINTS:
- Point 1
- Point 2  
- Point 3

CREDIBILITY: [Your assessment]
"""
        
        # Format and invoke
        formatted_prompt = prompt_template.format(
            topic=topic,
            url=url,
            title=title or "Untitled",
            content=content
        )
        
        response = llm.invoke(formatted_prompt)
        
        if hasattr(response, 'content'):
            response_text = response.content
        else:
            response_text = str(response)
        
        # Parse the simplified response
        key_points = []
        credibility_notes = "LLM-generated analysis"
        
        lines = response_text.split('\n')
        in_key_points = False
        
        for line in lines:
            line = line.strip()
            if 'KEY POINTS' in line.upper():
                in_key_points = True
                continue
            elif 'CREDIBILITY' in line.upper():
                in_key_points = False
                if ':' in line:
                    credibility_notes = line.split(':', 1)[1].strip()
                continue
            elif in_key_points and line.startswith('-'):
                key_points.append(line[1:].strip())
        
        if not key_points:
            # Extract any bullet points or numbered items
            for line in lines:
                if line.strip() and (line.startswith('•') or line.startswith('-') or line.startswith('*')):
                    key_points.append(line.strip().lstrip('•-*').strip())
        
        if not key_points:
            key_points = [f"Analysis of {topic} from this source"]
        
        from datetime import datetime
        return SourceSummary(
            source_id=source_id,
            url=url,
            title=title or "Untitled Source",
            key_points=key_points[:5],  # Limit to 5 key points
            credibility_notes=credibility_notes,
            extracted_at=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Failed to create source summary with LLM: {e}")
        from datetime import datetime
        return SourceSummary(
            source_id=source_id,
            url=url,
            title=title or "Untitled Source",
            key_points=[f"Failed to analyze content about {topic}"],
            credibility_notes=f"Error in analysis: {str(e)}",
            extracted_at=datetime.utcnow()
        )


def synthesize_final_brief_with_llm(
    topic: str,
    depth: int, 
    audience: str,
    plan: ResearchPlan,
    summaries: List[SourceSummary]
) -> FinalBrief:
    """Synthesize final brief using LLM or fallback to mock compilation."""
    manager = get_llm_manager()
    llm = manager.get_primary_llm()
    
    if not llm:
        logger.warning("No LLM available, using mock compilation")
        from models import compile_final_brief
        return compile_final_brief(topic, depth, audience, summaries)
    
    try:
        # Prepare concise context for free models
        plan_context = "Research Steps:\n"
        for step in plan.steps[:5]:  # Limit context length
            plan_context += f"{step.order}. {step.objective}\n"
        
        summaries_context = "Key Findings:\n"
        for i, summary in enumerate(summaries[:3], 1):  # Limit to 3 sources
            summaries_context += f"\nSource {i}: {summary.title}\n"
            summaries_context += f"Points: {'; '.join(summary.key_points[:3])}\n"
        
        # Simplified prompt for better results with free models
        prompt_template = """Create a research brief on: {topic}

Target Audience: {audience}
Depth Level: {depth}/5

{plan_context}

{summaries_context}

Create a comprehensive brief with:
1. A clear thesis statement
2. Well-structured sections with headings
3. Analysis of key findings
4. Relevant insights for {audience}

Focus on synthesis and analysis, not just summary.
"""
        
        formatted_prompt = prompt_template.format(
            topic=topic,
            depth=depth,
            audience=audience,
            plan_context=plan_context,
            summaries_context=summaries_context
        )
        
        response = llm.invoke(formatted_prompt)
        
        if hasattr(response, 'content'):
            response_text = response.content
        else:
            response_text = str(response)
        
        # Parse response into structured brief
        thesis = f"Research on {topic} reveals important insights and implications for {audience}."
        sections = []
        
        # Look for thesis
        lines = response_text.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in ['thesis:', 'main finding:', 'conclusion:']):
                thesis = line.split(':', 1)[-1].strip()
                break
        
        # Create sections from response
        current_section = None
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this looks like a section heading
            if _is_section_heading(line):
                # Save previous section
                if current_section and current_content:
                    from models import FinalBriefSection
                    sections.append(FinalBriefSection(
                        heading=current_section,
                        content='\n'.join(current_content)
                    ))
                
                # Start new section
                current_section = line.strip('#: ').title()
                current_content = []
            else:
                current_content.append(line)
        
        # Add final section
        if current_section and current_content:
            from models import FinalBriefSection
            sections.append(FinalBriefSection(
                heading=current_section,
                content='\n'.join(current_content)
            ))
        
        # Default sections if none found
        if not sections:
            from models import FinalBriefSection
            sections = [
                FinalBriefSection(
                    heading="Executive Summary",
                    content=response_text[:500] + "..." if len(response_text) > 500 else response_text
                )
            ]
        
        # Create references
        from models import FinalBriefReference
        references = [
            FinalBriefReference(
                source_id=summary.source_id,
                url=summary.url,
                title=summary.title
            ) for summary in summaries
        ]
        
        from datetime import datetime
        return FinalBrief(
            topic=topic,
            audience=audience,
            depth=depth,
            thesis=thesis,
            sections=sections,
            references=references,
            generated_at=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Failed to synthesize brief with LLM: {e}")
        logger.info("Falling back to mock compilation")
        from models import compile_final_brief
        return compile_final_brief(topic, depth, audience, summaries)


def _is_section_heading(line: str) -> bool:
    """Check if a line looks like a section heading."""
    line = line.strip()
    return (
        (line.isupper() and len(line.split()) <= 5) or
        (line.endswith(':') and len(line.split()) <= 5) or
        line.startswith('#') or
        (line.istitle() and len(line.split()) <= 4 and not line.endswith('.'))
    )


# Utility functions
def estimate_tokens(text: str) -> int:
    """Rough token estimation (4 characters per token)."""
    return len(text) // 4

def truncate_for_llm(text: str, max_tokens: int = 800) -> str:
    """Truncate text to fit within token limits (reduced for free models)."""
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "... [truncated]"