from pathlib import Path

MODEL_CONFIG_MAP = {

    # Automatically map all yml files from dfine config directory
    **{
        Path(f).name: str(
            Path("rtdetrv2") / Path(f).relative_to(Path(__file__).parent.parent.parent / "configs" / "yaml" / "rtdetrv2")
        )
        for f in (Path(__file__).parent.parent.parent / "configs" / "yaml" / "rtdetrv2").rglob("*.yml")
        if not Path(f).name.startswith("_")  # Skip include files
    },
}
