import json
for p in ["alfa.ipynb", "notebooks/colab_t4.ipynb"]:
    nb = json.load(open(p))
    src_all = ["".join(c.get("source", [])) for c in nb["cells"]]
    if any("userdata.get" in s for s in src_all):
        print(p, "already patched")
        continue
    login = {
        "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
        "source": [
            "from huggingface_hub import login\n",
            "from google.colab import userdata\n",
            "tok = userdata.get('HF_TOKEN')\n",
            "login(token=tok)\n",
            "print('logged in')\n",
        ],
    }
    nb["cells"].insert(1, login)
    for c in nb["cells"]:
        ns = []
        for line in c.get("source", []):
            if "snapshot_download" in line and "token=tok" not in line:
                line = line.replace("repo_type='model')", "repo_type='model', token=tok)")
            ns.append(line)
        c["source"] = ns
    json.dump(nb, open(p, "w"))
    print(p, "patched cells:", len(nb["cells"]))
