import argparse
from http import client
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yaml
from langchain_ollama import OllamaLLM as Ollama
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel

import openai
import anthropic
from google import genai


class RankedSong(BaseModel):
    rank: int
    uri: str
    song: str
    artist: str
    # score: float

class RankedSongsList(BaseModel):
    songs: List[RankedSong]  # Wrapper for multiple items
    
class RankedSongsListReasoning(BaseModel):
    songs: List[RankedSong]  # Wrapper for multiple items
    reasoning: Optional[str] = None  # Optional field for LLM to explain rankings

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"

DEFAULT_MODEL_IDS = {
    "gpt4o": "gpt-4o",
    "gpt41": "gpt-4.1",
    "gemini": "gemini-1.5-pro",
    "claude-latest": "claude-3.5-sonnet-latest",
}


def load_config(path: Optional[str] = None) -> dict:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def get_model_id(config: dict, model_key: str) -> str:
    return config.get("models", {}).get(model_key, DEFAULT_MODEL_IDS.get(model_key, model_key))


def call_openai(prompt_text: str, model_id: str, temperature: float, api_key: str) -> str:
    if openai is None:
        raise ImportError("openai is not installed. Install it to use OpenAI models.")
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt_text}],
        temperature=temperature,
        max_tokens=1200,
    )
    return response.choices[0].message.content or ""


def call_claude(prompt_text: str, model_id: str, temperature: float, api_key: str) -> str:
    if anthropic is None:
        raise ImportError("anthropic is not installed. Install it to use Claude models.")
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model_id,
        max_tokens=1200,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt_text}],
    )
    return response.content[0].text


def call_gemini(prompt_text: str, model_id: str, temperature: float, api_key: str) -> str:
    if genai is None:
        raise ImportError("google-genai is not installed. Install it to use Gemini models.")
    
    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=model_id,
        contents=prompt_text,
        temperature=temperature
    )

    return response.text or ""


# ------------------ CORE LIBRARY FUNCTION ------------------
def rank_playlist(
        title: str,
        songs_df: pd.DataFrame,
        prompt_text: str,
        model: str = "mistral",
        temperature: float = 0.2,
    config_path: Optional[str] = None,
    reasoning: bool = False,
) -> pd.DataFrame:
    """
    Re-rank songs based on relevance to a playlist title using PydanticOutputParser.

    Input DataFrame must have columns: uri, song, artist
    Returns DataFrame with columns: rank, uri, song, artist, score
    """

    # Validate input columns
    if not {"uri", "song", "artist"}.issubset(songs_df.columns):
        raise ValueError("songs_df must contain columns: uri, song, artist")

    # Convert songs to text list
    songs_text = "\n".join(f"- {row.uri} — {row.song} — {row.artist}"
                           for row in songs_df.itertuples(index=False))

    # Create Pydantic output parser
    parser_model = RankedSongsListReasoning if reasoning else RankedSongsList
    parser = PydanticOutputParser(pydantic_object=parser_model)
    format_instructions = parser.get_format_instructions()
    # print(format_instructions)

    # Prepare prompt
    if reasoning:
        prompt_text = (
            prompt_text
            + "\n\nSolve the following problem step by step and show your reasoning in the 'reasoning' property."
        )

    prompt = PromptTemplate(
        template=prompt_text,
        input_variables=["title", "songs"],
        partial_variables={"format_instructions": format_instructions},
    )
    rendered_prompt = prompt.format(title=title, songs=songs_text)
    # print(rendered_prompt)

    config = load_config(config_path)

    if model in {"gpt4o", "gpt41"}:
        api_key = config.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found in config.yaml or OPENAI_API_KEY.")
        model_id = get_model_id(config, model)
        response = call_openai(rendered_prompt, model_id, temperature, api_key)
    elif model == "gemini":
        api_key = config.get("google_api_key") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Google API key not found in config.yaml or GOOGLE_API_KEY.")
        model_id = get_model_id(config, model)
        response = call_gemini(rendered_prompt, model_id, temperature, api_key)
    elif model == "claude-latest":
        api_key = config.get("anthropic_api_key") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key not found in config.yaml or ANTHROPIC_API_KEY.")
        model_id = get_model_id(config, model)
        response = call_claude(rendered_prompt, model_id, temperature, api_key)
    else:
        llm = Ollama(
            model=model,
            temperature=temperature,
        )
        chain = prompt | llm
        response = chain.invoke({
            "title": title,
            "songs": songs_text,
        })

    print(response)

    if response.startswith('['):
        response = f'{{"songs": {response}}}' # Wrap in object for parser

    # Parse structured output
    parsed = parser.parse(response)  # RankedSongsList or RankedSongsListReasoning

    # print(parsed)
    # Convert to DataFrame
    df_ranked = pd.DataFrame([item.model_dump() for item in parsed.songs])

    # Sort by rank
    df_ranked = df_ranked.sort_values("rank").reset_index(drop=True)

    return df_ranked

# ------------------ CLI UTILITIES ------------------

def load_prompt(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def load_songs_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def main():
    parser = argparse.ArgumentParser(description="Re-rank playlist songs using an LLM")
    parser.add_argument("--title", required=True, help="Playlist title")
    parser.add_argument("--input", required=True, help="Input CSV (Song,Artist)")
    parser.add_argument("--output", required=True, help="Output CSV")
    parser.add_argument("--prompt", default='./prompt_ranker.txt', help="Prompt .txt file")
    parser.add_argument("--model", default="mistral")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--reasoning",
        action="store_true",
        help="Use RankedSongsListReasoning response schema",
    )

    args = parser.parse_args()

    songs_df = load_songs_csv(args.input)
    prompt_text = load_prompt(args.prompt)

    ranked_df = rank_playlist(
        title=args.title,
        songs_df=songs_df,
        prompt_text=prompt_text,
        model=args.model,
        temperature=args.temperature,
        reasoning=args.reasoning,
    )

    ranked_df.to_csv(args.output, index=False)
    print(f"Ranked playlist saved to {args.output}")


# ------------------ ENTRY POINT ------------------

if __name__ == "__main__":
    main()

# python ranker.py \
#   --title "K-pop" \
#   --input /Users/pasquale/git/LLM-Playlist-Recommender/similarity/output/top_songs_for_673925.csv \
#   --output ranked_673925.csv
