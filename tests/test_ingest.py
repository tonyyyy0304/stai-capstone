from scripts.ingest import ChangeSet, add_source_file, hash_source, normalize_text, plan_changes


def manifest_with(files):
    return {"files": {name: {"sha256": digest} for name, digest in files.items()}}


def test_plan_unchanged_is_noop():
    current = {"a.md": "hash-a", "b.md": "hash-b"}
    plan = plan_changes(current, manifest_with(current))
    assert plan == ChangeSet(to_ingest=[], unchanged=["a.md", "b.md"], deleted=[])


def test_plan_detects_new_changed_deleted():
    manifest = manifest_with({"a.md": "hash-a", "gone.md": "hash-g"})
    current = {"a.md": "hash-a-CHANGED", "new.md": "hash-n"}
    plan = plan_changes(current, manifest)
    assert plan.to_ingest == ["a.md", "new.md"]
    assert plan.unchanged == []
    assert plan.deleted == ["gone.md"]


def test_plan_force_reingests_everything():
    current = {"a.md": "hash-a"}
    plan = plan_changes(current, manifest_with(current), force=True)
    assert plan.to_ingest == ["a.md"]
    assert plan.unchanged == []


def test_plan_empty_manifest_ingests_all():
    plan = plan_changes({"a.md": "x"}, {"files": {}})
    assert plan.to_ingest == ["a.md"]


def test_normalize_fixes_hyphenation_and_page_artifacts():
    raw = "Employees are enti-\ntled to leave.\n\nPage 3 of 10\n\n\n\nNext   section.  \n"
    out = normalize_text(raw)
    assert "entitled" in out
    assert "Page 3" not in out
    assert "\n\n\n" not in out


def test_hash_source_includes_meta_sidecar(tmp_path):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-fake")
    before = hash_source(pdf)
    (tmp_path / "doc.meta.yaml").write_text("doc_id: doc\n", encoding="utf-8")
    after = hash_source(pdf)
    assert before != after  # editing the sidecar must trigger re-ingestion


def test_add_source_file_extends_frontmatter():
    doc = "---\ndoc_id: x\ntitle: X\ncategory: leave\n---\n\nBody text.\n"
    out = add_source_file(doc, "x.pdf")
    assert "source_file: x.pdf" in out.split("---")[1]
    assert out.endswith("Body text.\n")
