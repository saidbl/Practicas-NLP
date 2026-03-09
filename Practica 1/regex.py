import re

Match = ['wazzzzzup', 'wazzzup']
Skip = ['wazup', 'wazzup' ]

myRegEx = r"..zzz.."

GREEN = "\033[92m"
RESET = "\033[0m"

def highlight_match(text, regex):
    match = re.search(regex, text)
    if match:
        start, end = match.span()
        highlighted = (
            text[:start] +
            GREEN + text[start:end] + RESET +
            text[end:]
        )
        return highlighted
    return text
print("Match:")
for word in Match:
    highlighted = highlight_match(word, myRegEx)
    print(f"{word} : {highlighted}")

print("\nSkip:")
for word in Skip:
    highlighted = highlight_match(word, myRegEx)
    print(f"{word}:{highlighted}")