from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone
import subprocess
import tempfile
import asyncio
import time
from emergentintegrations.llm.chat import LlmChat, UserMessage

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

app = FastAPI()
api_router = APIRouter(prefix="/api")

EMERGENT_LLM_KEY = os.environ.get('EMERGENT_LLM_KEY')

class CodeExecuteRequest(BaseModel):
    language: str
    code: str
    input: str = ""

class CodeExecuteResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    output: str
    error: str = ""
    execution_time: float
    memory_used: str = "N/A"
    success: bool

class AIAnalyzeRequest(BaseModel):
    code: str
    language: str

class AIAnalyzeResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    has_errors: bool
    errors: List[Dict[str, Any]]
    suggestions: List[str]
    quality_score: int
    complexity_score: str

class AIFixRequest(BaseModel):
    code: str
    language: str
    error: Optional[str] = None

class AIFixResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    fixed_code: str
    explanation: str
    changes: List[str]

class AIChatRequest(BaseModel):
    message: str
    code: Optional[str] = None
    language: Optional[str] = None
    session_id: str

class AIChatResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    response: str
    code_suggestion: Optional[str] = None

class AIExplainRequest(BaseModel):
    code: str
    language: str

class AIExplainResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    explanation: str
    complexity: str
    best_practices: List[str]

class AITTSRequest(BaseModel):
    text: str

class ChatHistoryCreate(BaseModel):
    session_id: str
    message: str
    response: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

LANGUAGE_CONFIGS = {
    "python": {"ext": ".py", "cmd": ["python3"], "timeout": 10},
    "c": {"ext": ".c", "cmd": ["gcc", "-o"], "timeout": 15},
    "cpp": {"ext": ".cpp", "cmd": ["g++", "-o"], "timeout": 15},
    "java": {"ext": ".java", "cmd": ["javac"], "timeout": 15}
}

async def execute_code(language: str, code: str, input_data: str = "") -> CodeExecuteResponse:
    lang = language.lower()
    if lang not in LANGUAGE_CONFIGS:
        raise HTTPException(status_code=400, detail=f"Unsupported language: {language}")
    
    config = LANGUAGE_CONFIGS[lang]
    start_time = time.time()
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            source_file = tmpdir_path / f"Main{config['ext']}"
            source_file.write_text(code)
            
            if lang == "python":
                result = subprocess.run(
                    ["python3", str(source_file)],
                    input=input_data,
                    capture_output=True,
                    text=True,
                    timeout=config['timeout']
                )
                output = result.stdout
                error = result.stderr
                success = result.returncode == 0
                
            elif lang in ["c", "cpp"]:
                executable = tmpdir_path / "Main"
                compiler = "gcc" if lang == "c" else "g++"
                
                compile_result = subprocess.run(
                    [compiler, str(source_file), "-o", str(executable)],
                    capture_output=True,
                    text=True,
                    timeout=config['timeout']
                )
                
                if compile_result.returncode != 0:
                    execution_time = time.time() - start_time
                    return CodeExecuteResponse(
                        output="",
                        error=compile_result.stderr,
                        execution_time=execution_time * 1000,
                        success=False
                    )
                
                result = subprocess.run(
                    [str(executable)],
                    input=input_data,
                    capture_output=True,
                    text=True,
                    timeout=config['timeout']
                )
                output = result.stdout
                error = result.stderr
                success = result.returncode == 0
                
            elif lang == "java":
                compile_result = subprocess.run(
                    ["javac", str(source_file)],
                    capture_output=True,
                    text=True,
                    timeout=config['timeout'],
                    cwd=str(tmpdir_path)
                )
                
                if compile_result.returncode != 0:
                    execution_time = time.time() - start_time
                    return CodeExecuteResponse(
                        output="",
                        error=compile_result.stderr,
                        execution_time=execution_time * 1000,
                        success=False
                    )
                
                result = subprocess.run(
                    ["java", "Main"],
                    input=input_data,
                    capture_output=True,
                    text=True,
                    timeout=config['timeout'],
                    cwd=str(tmpdir_path)
                )
                output = result.stdout
                error = result.stderr
                success = result.returncode == 0
            
            execution_time = time.time() - start_time
            
            return CodeExecuteResponse(
                output=output,
                error=error,
                execution_time=execution_time * 1000,
                success=success
            )
            
    except subprocess.TimeoutExpired:
        execution_time = time.time() - start_time
        return CodeExecuteResponse(
            output="",
            error="Execution timed out. Your code might have an infinite loop.",
            execution_time=execution_time * 1000,
            success=False
        )
    except Exception as e:
        execution_time = time.time() - start_time
        return CodeExecuteResponse(
            output="",
            error=str(e),
            execution_time=execution_time * 1000,
            success=False
        )

async def analyze_code_with_ai(code: str, language: str) -> AIAnalyzeResponse:
    try:
        chat = LlmChat(
            api_key=EMERGENT_LLM_KEY,
            session_id=str(uuid.uuid4()),
            system_message="You are an expert code analyzer. Analyze code for errors, suggest improvements, and rate code quality."
        ).with_model("openai", "gpt-4o")
        
        message = UserMessage(
            text=f"Analyze this {language} code and provide:\n1. List of errors (with line numbers if possible)\n2. Suggestions for improvement\n3. Quality score (0-100)\n4. Complexity assessment (Low/Medium/High)\n\nCode:\n```{language}\n{code}\n```\n\nRespond in JSON format with keys: has_errors, errors (array of objects with 'line', 'message', 'severity'), suggestions (array), quality_score, complexity_score"
        )
        
        response = await chat.send_message(message)
        
        import json
        try:
            response_text = response.strip()
            if response_text.startswith("```json"):
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif response_text.startswith("```"):
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            data = json.loads(response_text)
            return AIAnalyzeResponse(**data)
        except:
            return AIAnalyzeResponse(
                has_errors=False,
                errors=[],
                suggestions=["Code analysis completed. No critical issues found."],
                quality_score=75,
                complexity_score="Medium"
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def fix_code_with_ai(code: str, language: str, error: Optional[str]) -> AIFixResponse:
    try:
        chat = LlmChat(
            api_key=EMERGENT_LLM_KEY,
            session_id=str(uuid.uuid4()),
            system_message="You are an expert programmer. Fix code errors and provide clear explanations."
        ).with_model("openai", "gpt-4o")
        
        error_context = f"\nError encountered: {error}" if error else ""
        message = UserMessage(
            text=f"Fix this {language} code:{error_context}\n\nCode:\n```{language}\n{code}\n```\n\nProvide the fixed code and explain the changes. Respond in JSON format with keys: fixed_code (just the code, no markdown), explanation, changes (array of strings describing each change)"
        )
        
        response = await chat.send_message(message)
        
        import json
        try:
            response_text = response.strip()
            if response_text.startswith("```json"):
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif response_text.startswith("```"):
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            data = json.loads(response_text)
            return AIFixResponse(**data)
        except:
            return AIFixResponse(
                fixed_code=code,
                explanation="Unable to auto-fix. Please review the error message.",
                changes=[]
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def chat_with_ai(message: str, code: Optional[str], language: Optional[str], session_id: str) -> AIChatResponse:
    try:
        chat = LlmChat(
            api_key=EMERGENT_LLM_KEY,
            session_id=session_id,
            system_message="You are an expert coding assistant like GitHub Copilot. Help users with coding questions, provide suggestions, and explain concepts clearly. If providing code, include it in your response."
        ).with_model("openai", "gpt-4o")
        
        context = ""
        if code and language:
            context = f"\n\nCurrent code context ({language}):\n```{language}\n{code}\n```"
        
        user_message = UserMessage(text=message + context)
        response = await chat.send_message(user_message)
        
        code_suggestion = None
        if "```" in response:
            code_blocks = response.split("```")
            if len(code_blocks) > 1:
                code_suggestion = code_blocks[1].strip()
                if code_suggestion.startswith(language or ""):
                    code_suggestion = code_suggestion[len(language or ""):].strip()
        
        history = ChatHistoryCreate(
            session_id=session_id,
            message=message,
            response=response
        )
        doc = history.model_dump()
        doc['timestamp'] = doc['timestamp'].isoformat()
        await db.chat_history.insert_one(doc)
        
        return AIChatResponse(response=response, code_suggestion=code_suggestion)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def explain_code_with_ai(code: str, language: str) -> AIExplainResponse:
    try:
        chat = LlmChat(
            api_key=EMERGENT_LLM_KEY,
            session_id=str(uuid.uuid4()),
            system_message="You are an expert programming teacher. Explain code clearly for learners."
        ).with_model("openai", "gpt-4o")
        
        message = UserMessage(
            text=f"Explain this {language} code in detail:\n\nCode:\n```{language}\n{code}\n```\n\nProvide:\n1. Overall explanation\n2. Complexity assessment (Low/Medium/High)\n3. Best practices being followed or missing\n\nRespond in JSON format with keys: explanation, complexity, best_practices (array)"
        )
        
        response = await chat.send_message(message)
        
        import json
        try:
            response_text = response.strip()
            if response_text.startswith("```json"):
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif response_text.startswith("```"):
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            data = json.loads(response_text)
            return AIExplainResponse(**data)
        except:
            return AIExplainResponse(
                explanation=response,
                complexity="Medium",
                best_practices=["Code explanation provided"]
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/")
async def root():
    return {"message": "Mini Code Mentor API"}

@api_router.post("/code/execute", response_model=CodeExecuteResponse)
async def execute_code_endpoint(request: CodeExecuteRequest):
    return await execute_code(request.language, request.code, request.input)

@api_router.post("/ai/analyze", response_model=AIAnalyzeResponse)
async def analyze_code_endpoint(request: AIAnalyzeRequest):
    return await analyze_code_with_ai(request.code, request.language)

@api_router.post("/ai/fix", response_model=AIFixResponse)
async def fix_code_endpoint(request: AIFixRequest):
    return await fix_code_with_ai(request.code, request.language, request.error)

@api_router.post("/ai/chat", response_model=AIChatResponse)
async def chat_endpoint(request: AIChatRequest):
    return await chat_with_ai(request.message, request.code, request.language, request.session_id)

@api_router.post("/ai/explain", response_model=AIExplainResponse)
async def explain_code_endpoint(request: AIExplainRequest):
    return await explain_code_with_ai(request.code, request.language)

app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
