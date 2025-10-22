import os
import json
import argparse
import glob
import time
from datetime import datetime
import openai
import pandas as pd
from dotenv import load_dotenv
from pypdf import PdfReader
from agent import AnalysisAgent


def main():
    parser = argparse.ArgumentParser(description="Run CellVoyager analysis agent")
    
    # REQUIRED data argument
    parser.add_argument("--data-path",
                       default=None,
                       help="Path to tabular dataset (csv, parquet, feather, dta). Required.")

    # Replication directory mode (optional)
    parser.add_argument("--data-dir",
                       default=None,
                       help="Directory containing multiple data files (replication package). If set, overrides --data-path.")
    parser.add_argument("--data-glob",
                       default="*.csv,*.parquet,*.feather,*.dta",
                       help="Comma-separated patterns to discover tabular files under --data-dir (default: *.csv,*.parquet,*.feather,*.dta)")
    parser.add_argument("--primary-file",
                       default=None,
                       help="Optional explicit primary file within --data-dir to use (path or basename)")
    parser.add_argument("--catalog-only",
                       action="store_true",
                       help="If set with --data-dir, only build and save a dataset catalog and exit")
    
    parser.add_argument("--paper-pdf",
                       required=True,
                       help="Path to research paper PDF (required); text is extracted and summarized automatically")
    
    parser.add_argument("--analysis-name", 
                       default="covid19",
                       help="Name for the analysis (default: covid19)")
    
    # Optional arguments with defaults
    parser.add_argument("--model-name", 
                       default="o3-mini",
                       help="OpenAI model name to use (default: o3-mini)")
    
    parser.add_argument("--num-analyses", 
                       type=int, 
                       default=8,
                       help="Number of analyses to run (default: 8)")
    
    parser.add_argument("--max-iterations", 
                       type=int, 
                       default=6,
                       help="Maximum iterations per analysis (default: 6)")
    
    parser.add_argument("--max-fix-attempts", 
                       type=int, 
                       default=3,
                       help="Maximum fix attempts per step (default: 3)")
    
    parser.add_argument("--output-home", 
                       default=".",
                       help="Home directory for outputs (default: current directory)")
    
    parser.add_argument("--log-home", 
                       default=".",
                       help="Home directory for logs (default: current directory)")
    
    parser.add_argument("--prompt-dir", 
                       default="prompts",
                       help="Directory containing prompt templates (default: prompts)")
    
    # Boolean flags
    parser.add_argument("--no-self-critique", 
                       action="store_true",
                       help="Disable self-critique functionality")
    
    parser.add_argument("--no-vlm", 
                       action="store_true",
                       help="Disable Vision Language Model functionality")
    
    parser.add_argument("--no-documentation", 
                       action="store_true",
                       help="Disable documentation functionality")
    
    parser.add_argument("--log-prompts", 
                       action="store_true",
                       help="Enable prompt logging")

    # Optional schema/context for econ datasets
    parser.add_argument("--outcome", default=None, help="Outcome variable column name")
    parser.add_argument("--treatment", default=None, help="Treatment indicator column name")
    parser.add_argument("--time-var", default=None, help="Time variable column name for panels/event studies")
    parser.add_argument("--unit-var", default=None, help="Unit identifier column name for panels")
    parser.add_argument("--cluster-se", default=None, help="Column name for clustering standard errors")
    
    args = parser.parse_args()
    # Load environment variables from .env if present
    load_dotenv()
    
    # Check if OpenAI API key is available
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key: export OPENAI_API_KEY='your-key-here'")
        return 1
    
    # Extract and summarize the paper PDF
    if not os.path.exists(args.paper_pdf):
        print(f"❌ Error: Paper PDF not found: {args.paper_pdf}")
        return 1
    
    try:
            print("📄 Extracting text from paper PDF...")
            reader = PdfReader(args.paper_pdf)
            raw_text = []
            for page in reader.pages:
                try:
                    raw_text.append(page.extract_text() or "")
                except Exception:
                    continue
            paper_text = "\n\n".join(raw_text).strip()
            if not paper_text:
                print("❌ Failed to extract any text from PDF")
                return 1

            # Summarize with OpenAI into a concise empirical-econ summary
            # Always use a fast model for summarization (not o3)
            print("🧾 Summarizing paper content with LLM...")
            client = openai.OpenAI(api_key=openai_api_key)
            summary_model = "gpt-4o-mini"  # Fast and cheap for summarization
            summary_prompt = (
                "You are an empirical economics assistant. Summarize the paper succinctly, focusing on: "
                "research question, dataset(s), key variables (likely outcome, treatment, time, unit), "
                "identification strategy (e.g., OLS/FE, DID/event-study, IV), and any notable caveats. "
                "Return a clear 10-15 sentence summary that can guide downstream analysis.\n\n"
                "Paper text begins:\n" + paper_text[:120000]  # cap to avoid excessive tokens
            )
            resp = client.chat.completions.create(
                model=summary_model,
                messages=[
                    {"role": "system", "content": "You write concise, structured empirical economics summaries."},
                    {"role": "user", "content": summary_prompt},
                ],
            )
            paper_summary_txt = resp.choices[0].message.content or ""
            if not paper_summary_txt.strip():
                paper_summary_txt = paper_text[:5000]

            os.makedirs("logs", exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            extracted_summary_path = os.path.join("logs", f"paper_summary_extracted_{ts}.txt")
            with open(extracted_summary_path, "w") as f:
                f.write(paper_summary_txt)
            print(f"📝 Saved extracted paper summary → {extracted_summary_path}")
            paper_summary_path = extracted_summary_path
    except Exception as e:
        print(f"❌ Error extracting/summarizing PDF: {e}")
        return 1
    
    # Resolve dataset source
    selected_data_path = None
    catalog = None

    if args.data_dir:
        if not os.path.isdir(args.data_dir):
            print(f"❌ Error: Data directory not found: {args.data_dir}")
            return 1

        # Discover files by patterns
        patterns = [p.strip() for p in (args.data_glob or "").split(",") if p.strip()]
        files = []
        for pat in patterns:
            files.extend(glob.glob(os.path.join(args.data_dir, "**", pat), recursive=True))
        files = sorted(set(files))
        if not files:
            print(f"❌ No tabular files found in {args.data_dir} with patterns {patterns}")
            return 1

        # Build lightweight catalog
        def infer_file(fp):
            ext = os.path.splitext(fp)[1].lower()
            size = os.path.getsize(fp)
            cols, nrows, ncols = [], None, None
            sample_cols = []
            try:
                if ext == ".csv":
                    df = pd.read_csv(fp, nrows=200)
                elif ext == ".parquet":
                    df = pd.read_parquet(fp)
                elif ext == ".feather":
                    df = pd.read_feather(fp)
                elif ext == ".dta":
                    # Stata files can be slow; only read first 200 rows for cataloging
                    iter_dta = pd.read_stata(fp, chunksize=200, convert_categoricals=False)
                    df = next(iter_dta)
                else:
                    return {"path": fp, "ext": ext, "size": size, "error": "unsupported_ext"}
                nrows, ncols = df.shape
                cols = list(df.columns)
                sample_cols = cols[:20]
            except Exception as e:
                return {"path": fp, "ext": ext, "size": size, "error": str(e)}

            # Heuristic signals
            colset = {c.lower(): c for c in cols}
            has_time = any(k in colset for k in ["time", "year", "date", "t"])
            has_unit = any(k in colset for k in ["unit", "id", "panelid", "county", "state", "region", "firm"])
            has_treat = any(k in colset for k in ["treat", "treated", "policy", "post", "d"])
            has_outcome = any(k in colset for k in ["y", "outcome", "dep", "dependent", "lhs"])

            score = 0
            score += 3 if has_unit and has_time else 0
            score += 2 if has_treat else 0
            score += 1 if has_outcome else 0
            score += 0.5 * (nrows or 0) / 1_000_000 + 0.1 * (ncols or 0)  # tiny tie-breaker by size

            return {
                "path": fp, "ext": ext, "size": size, "nrows": nrows, "ncols": ncols,
                "columns": sample_cols, "score": score,
                "signals": {"has_unit": has_unit, "has_time": has_time, "has_treat": has_treat, "has_outcome": has_outcome}
            }

        catalog = [infer_file(fp) for fp in files]

        # Choose primary file
        if args.primary_file:
            # Allow basename or full path
            candidate = args.primary_file
            matched = [c for c in catalog if c["path"] == candidate or os.path.basename(c["path"]) == os.path.basename(candidate)]
            if not matched:
                print(f"❌ --primary-file not found in catalog: {args.primary_file}")
                return 1
            selected_data_path = matched[0]["path"]
        else:
            # Highest score wins; tie-break by nrows then size
            valid = [c for c in catalog if not c.get("error")]
            if not valid:
                print("❌ No readable tabular files found in directory")
                return 1
            valid.sort(key=lambda c: (c.get("score", 0), c.get("nrows") or 0, c.get("size") or 0), reverse=True)
            selected_data_path = valid[0]["path"]

        # Save catalog
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("logs", exist_ok=True)
        cat_path = os.path.join("logs", f"dataset_catalog_{ts}.json")
        try:
            with open(cat_path, 'w') as f:
                json.dump({
                    "data_dir": os.path.abspath(args.data_dir),
                    "patterns": patterns,
                    "selected": selected_data_path,
                    "files": catalog
                }, f, indent=2)
            print(f"🗂  Saved dataset catalog → {cat_path}")
        except Exception:
            pass

        if args.catalog_only:
            print("Catalog-only mode: exiting without running analyses.")
            return 0
    else:
        # Single file mode
        if not args.data_path or not os.path.exists(args.data_path):
            print(f"❌ Error: Data file not found: {args.data_path}")
            return 1
        selected_data_path = args.data_path
    
    print("🚀 Starting AEAExtensions Analysis Agent")
    print(f"   Data file: {selected_data_path}")
    print(f"   Paper summary: {paper_summary_path}")
    print(f"   Analysis name: {args.analysis_name}")
    print(f"   Model: {args.model_name}")
    print(f"   Number of analyses: {args.num_analyses}")
    print(f"   Max iterations: {args.max_iterations}")
    print(f"   Self-critique: {'❌' if args.no_self_critique else '✅'}")
    print(f"   VLM: {'❌' if args.no_vlm else '✅'}")
    print(f"   Documentation: {'❌' if args.no_documentation else '✅'}")
    print()
    
    # Initialize the agent
    agent = AnalysisAgent(
        data_path=selected_data_path,
        paper_summary_path=paper_summary_path,
        openai_api_key=openai_api_key,
        model_name=args.model_name,
        analysis_name=args.analysis_name,
        num_analyses=args.num_analyses,
        max_iterations=args.max_iterations,
        prompt_dir=args.prompt_dir,
        output_home=args.output_home,
        log_home=args.log_home,
        use_self_critique=not args.no_self_critique,
        use_VLM=not args.no_vlm,
        use_documentation=not args.no_documentation,
        log_prompts=args.log_prompts,
        max_fix_attempts=args.max_fix_attempts,
        
        # Pass schema context (may be None)
        outcome_var=args.outcome,
        treatment_var=args.treatment,
        time_var=args.time_var,
        unit_var=args.unit_var,
        cluster_se_var=args.cluster_se
    )
    
    try:
        # Run the analysis
        print("🔬 Running analyses...")
        agent.run()
        print("✅ Analysis complete!")
            
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️ Analysis interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return 1
    finally:
        # Clean up agent resources
        if hasattr(agent, 'cleanup'):
            agent.cleanup()


if __name__ == "__main__":
    exit(main())
