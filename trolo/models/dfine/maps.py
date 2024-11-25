from pathlib import Path

MODEL_CONFIG_MAP = {
    # Base model names (short aliases)
    "dfine-n": "dfine/dfine_hgnetv2_n_coco.yml",
    "dfine-s": "dfine/dfine_hgnetv2_s_coco.yml", 
    "dfine-m": "dfine/dfine_hgnetv2_m_coco.yml",
    "dfine-l": "dfine/dfine_hgnetv2_l_coco.yml",
    "dfine-x": "dfine/dfine_hgnetv2_x_coco.yml",

    # pth # TODO this won't be needed as models store configs
    "dfine_n_coco.pth": "dfine/dfine_hgnetv2_n_coco.yml",
    "dfine_s_coco.pth": "dfine/dfine_hgnetv2_s_coco.yml",
    "dfine_m_coco.pth": "dfine/dfine_hgnetv2_m_coco.yml",
    "dfine_l_coco.pth": "dfine/dfine_hgnetv2_l_coco.yml",
    "dfine_x_coco.pth": "dfine/dfine_hgnetv2_x_coco.yml",


    # Automatically map all yml files from dfine config directory
    **{
        Path(f).name: str(Path('dfine') / Path(f).relative_to(Path(__file__).parent.parent.parent / 'configs' / 'yaml' / 'dfine'))
        for f in (Path(__file__).parent.parent.parent / 'configs' / 'yaml' / 'dfine').rglob('*.yml')
        if not Path(f).name.startswith('_')  # Skip include files
    },


}