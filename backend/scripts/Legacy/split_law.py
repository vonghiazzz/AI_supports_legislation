import json
import re
import os

def split_laws():
    # List of OCR files that have been run
    sources = ["scripts/output.json", "scripts/output1.json"]
    all_articles = []

    for src in sources:
        if os.path.exists(src):
            print(f"Loading {src}...")
            file_text = ""
            with open(src, "r", encoding="utf-8") as f:
                data = json.load(f)
                for page in data:
                    file_text += page.get("text", "") + "\n"
            
            # Normalize common OCR errors at the start of lines (e.g., ",Điều 202.") 
            file_text = re.sub(r'\n[^\w\n]+(Điều\s+)', r'\n\1', '\n' + file_text)
            
            # Split by any "Điều " string at the start of a line to process
            blocks = re.split(r'\n(?=Điều\s+)', file_text)
            articles_in_file = []
            current_art = ""
            last_id = 0
            
            for block in blocks:
                block = block.strip()
                if not block: continue
                
                is_new_article = False
                law_id = 0
                
                # Match strict Article with number
                match_strict = re.match(r'^Điều\s+(\d+)\.', block)
                if match_strict:
                    law_id = int(match_strict.group(1))
                    is_new_article = True
                else:
                    # Match Article with OCR errors (missing dots, or letters replacing numbers like "Ra")
                    match_loose = re.match(r'^Điều\s+([A-Za-z0-9]+)\b', block)
                    if match_loose:
                        token = match_loose.group(1)
                        if token.isdigit() and int(token) == last_id + 1:
                            is_new_article = True
                            law_id = int(token)
                        elif not token.isdigit() and token[0].isupper():
                            # If this is an uppercase letter likely to be an OCR error from a number, and not a citation line
                            first_line = block.split('\n')[0]
                            if "của Bộ luật" not in first_line and "khoản" not in first_line:
                                is_new_article = True
                                law_id = last_id + 1
                                
                if is_new_article:
                    if current_art:
                        articles_in_file.append(current_art)
                    # Reattach the correct format "Điều <id>." if taken from OCR error
                    if not match_strict:
                        block = re.sub(r'^Điều\s+[A-Za-z0-9]+\b[^\w]*', f'Điều {law_id}. ', block, count=1)
                    current_art = "\n" + block
                    last_id = law_id
                else:
                    if current_art:
                        current_art += "\n\n" + block

            if current_art:
                articles_in_file.append(current_art)
            filename = os.path.basename(src) # "output.json" or "output1.json"
            # Note: User wants the original PDF file name (91_.pdf / 91.signed.pdf)
            # Base on meta in output.json (if any) or manual mapping
            original_file = "91_.pdf" if src == "scripts/output.json" else "91.signed.pdf"

            for i, art in enumerate(articles_in_file):
                content = art.strip()
                match_id = re.search(r"Điều\s+(\d+)", content)
                law_id = match_id.group(1) if match_id else None
                
                if law_id:
                    all_articles.append({
                        "law_name": "Luật Dân Sự 2015",
                        "law_id": law_id,
                        "content": content,
                        "source": original_file
                    })

    # Deduplicate: If an Article appears in both files, take the version from output.json (usually cleaner)
    unique_articles = {}
    for art in all_articles:
        lid = art["law_id"]
        if lid not in unique_articles:
            unique_articles[lid] = art
    
    result = list(unique_articles.values())
    # Sort by law_id (numeric)
    result.sort(key=lambda x: int(x["law_id"]) if x["law_id"].isdigit() else 9999)

    with open("law_articles.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"✅ Successfully extracted {len(result)} unique articles to law_articles.json")

if __name__ == "__main__":
    split_laws()