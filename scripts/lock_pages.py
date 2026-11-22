"""Quarto post-render step: password-protect the hands-on part of nuclear.html.

The element <div id="locked-content"> … </div> of _site/nuclear.html is encrypted with
AES-256-GCM (key derived from the password with PBKDF2-SHA256) and replaced by a small form;
the browser decrypts it with the Web Crypto API when the right password is typed.

The password is read from the environment variable NUCLEAR_PASSWORD (a GitHub secret in the
publish workflow). If it is not set, the content is simply removed: the page fails closed.
The page is also removed from the site search index.

Limits (honest): this hides the *page*. The notebook files themselves are in the public GitHub
repository; someone who browses the repository can find them.
"""
import base64
import json
import os
import re
import secrets
import sys
from pathlib import Path

SITE = Path(os.environ.get("QUARTO_PROJECT_OUTPUT_DIR", "_site"))
PAGE = SITE / "nuclear.html"
ITERATIONS = 250_000
MARK = 'id="locked-content"'


def find_div(html):
    """Return (start, end) of the element carrying id="locked-content", end exclusive."""
    i = html.find(MARK)
    if i < 0:
        return None
    start = html.rfind("<", 0, i)
    name = re.match(r"<(\w+)", html[start:]).group(1)          # div (or section)
    depth = 0
    tag = re.compile(rf"<(/?){name}\b[^>]*>", re.I)
    for m in tag.finditer(html, start):
        depth += -1 if m.group(1) else 1
        if depth == 0:
            return start, m.end()
    raise RuntimeError(f"unbalanced <{name}> in nuclear.html")


def encrypt(plaintext, password):
    try:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    except ImportError:
        sys.exit("lock_pages.py: NUCLEAR_PASSWORD is set but the 'cryptography' package is missing "
                 "(pip install cryptography)")
    salt, iv = secrets.token_bytes(16), secrets.token_bytes(12)
    key = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt,
                     iterations=ITERATIONS).derive(password.encode())
    ct = AESGCM(key).encrypt(iv, plaintext.encode(), None)        # ciphertext ‖ tag
    b64 = lambda b: base64.b64encode(b).decode()
    return {"salt": b64(salt), "iv": b64(iv), "ct": b64(ct), "iter": ITERATIONS}


FORM = """<div id="locked-content" class="locked-box">
  <p class="locked-title">🔒 Hands-on notebooks — unlocked during the workshop</p>
  <p>The three nuclear-energy notebooks (radionuclide identification, reactor-physics surrogate,
  neutron/gamma pulse-shape discrimination) open with the password given in the session.
  In the meantime, start with the <a href="exercises.html">accelerator exercises</a>.</p>
  {form}
</div>"""

FORM_FIELDS = """<form id="unlock-form" autocomplete="off">
    <input id="unlock-pw" type="password" placeholder="password" aria-label="password">
    <button type="submit">Unlock</button>
    <span id="unlock-msg" class="locked-msg"></span>
  </form>
  <script id="locked-data" type="application/json">{payload}</script>
  <script>
  (function () {{
    const data = JSON.parse(document.getElementById("locked-data").textContent);
    const box = document.getElementById("locked-content");
    const msg = document.getElementById("unlock-msg");
    const b = s => Uint8Array.from(atob(s), c => c.charCodeAt(0));
    async function unlock(pw) {{
      const base = await crypto.subtle.importKey("raw", new TextEncoder().encode(pw),
                                                 "PBKDF2", false, ["deriveKey"]);
      const key = await crypto.subtle.deriveKey(
        {{name: "PBKDF2", salt: b(data.salt), iterations: data.iter, hash: "SHA-256"}},
        base, {{name: "AES-GCM", length: 256}}, false, ["decrypt"]);
      const plain = await crypto.subtle.decrypt({{name: "AES-GCM", iv: b(data.iv)}}, key, b(data.ct));
      const tmp = document.createElement("div");
      tmp.innerHTML = new TextDecoder().decode(plain);
      box.replaceWith(tmp.firstElementChild);
      try {{ sessionStorage.setItem("ml4acc-nuclear", pw); }} catch (e) {{}}
      if (location.hash) {{ const t = document.querySelector(location.hash); if (t) t.scrollIntoView(); }}
    }}
    document.getElementById("unlock-form").addEventListener("submit", async ev => {{
      ev.preventDefault();
      msg.textContent = "…";
      try {{ await unlock(document.getElementById("unlock-pw").value); }}
      catch (e) {{ msg.textContent = "Wrong password."; }}
    }});
    let saved = null;
    try {{ saved = sessionStorage.getItem("ml4acc-nuclear"); }} catch (e) {{}}
    if (saved) unlock(saved).catch(() => {{}});
  }})();
  </script>"""

NO_PASSWORD = "<p><em>(This build was made without a password: the notebooks are not included.)</em></p>"


def scrub_search():
    idx = SITE / "search.json"
    if idx.exists():
        entries = json.loads(idx.read_text())
        kept = [e for e in entries if not e.get("href", "").startswith("nuclear.html")]
        idx.write_text(json.dumps(kept))


def main():
    if not PAGE.exists():
        return
    html = PAGE.read_text()
    if 'class="locked-box"' in html:
        print("lock_pages.py: nuclear.html already locked")
        scrub_search()
        return
    span = find_div(html)
    if span is None:
        print("lock_pages.py: no locked content found (already locked?)")
        return
    s, e = span
    password = os.environ.get("NUCLEAR_PASSWORD", "")
    if password:
        payload = json.dumps(encrypt(html[s:e], password)).replace("</", "<\\/")
        block = FORM.replace("{form}", FORM_FIELDS.format(payload=payload))
        status = "encrypted"
    else:
        block = FORM.replace("{form}", NO_PASSWORD)
        status = "removed (NUCLEAR_PASSWORD not set)"
    PAGE.write_text(html[:s] + block + html[e:])
    scrub_search()
    print(f"lock_pages.py: nuclear.html locked content {status}")


if __name__ == "__main__":
    main()
