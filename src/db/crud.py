from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from src.db.models import User, Dataset, ChatSession, Message, AgentRun

def get_or_create_user(db: Session, user_id: str):
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        user = User(user_id=user_id)
        db.add(user)
        try:
            db.commit()
            db.refresh(user)
        except IntegrityError:
            db.rollback()
            user = db.query(User).filter(User.user_id == user_id).first()
    return user

def create_dataset(db: Session, user_id: str, filename: str, collection_name: str, provider: str, data_mode: str, chunks_count: int = 0, summary: str = ""):
    # Ensure user exists first
    get_or_create_user(db, user_id)
    
    dataset = Dataset(
        user_id=user_id,
        filename=filename,
        collection_name=collection_name,
        provider=provider,
        data_mode=data_mode,
        chunks_count=chunks_count,
        summary=summary
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return dataset

def get_dataset_by_collection(db: Session, collection_name: str):
    return db.query(Dataset).filter(Dataset.collection_name == collection_name).first()

def create_session(db: Session, user_id: str, dataset_id: int = None):
    # Ensure user exists first
    get_or_create_user(db, user_id)
    
    session = ChatSession(
        user_id=user_id,
        dataset_id=dataset_id
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session

def log_message(db: Session, session_id: int, role: str, content: str, tokens_used: int = 0):
    message = Message(
        session_id=session_id,
        role=role,
        content=content,
        tokens_used=tokens_used
    )
    db.add(message)
    db.commit()
    db.refresh(message)
    return message

def log_agent_run(db: Session, session_id: int, node_name: str, input_data: str, output_data: str, latency_ms: float, status: str):
    # Truncate previews to 1000 characters to avoid huge string insertions if they are massive
    input_preview = str(input_data)[:1000] if input_data else ""
    output_preview = str(output_data)[:1000] if output_data else ""
    
    run = AgentRun(
        session_id=session_id,
        node_name=node_name,
        input_preview=input_preview,
        output_preview=output_preview,
        latency_ms=latency_ms,
        status=status
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run
