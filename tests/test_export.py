import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from trolo.export import ModelExporter
from trolo.utils.smart_defaults import infer_pretrained_model

DEFAULT_MODEL = "dfine_n.pth"

@pytest.fixture(scope="session")
def model_path():
    return infer_pretrained_model(DEFAULT_MODEL)

@pytest.fixture
def exporter(model_path):
    return ModelExporter(DEFAULT_MODEL)

@pytest.fixture
def create_temp_dir():
    with TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        yield temp_dir

def test_onnx_export(exporter, tmp_path):
    input_size = (640, 640)
    exporter.export(
        export_format='onnx', 
        input_size=input_size
    )
    assert Path(DEFAULT_MODEL).with_suffix('.onnx').exists()

@pytest.mark.parametrize("input_size", [
    (640, 640)
])
def test_export_variable_input_sizes(exporter, input_size):
    exporter.export(
        export_format='onnx', 
        input_size=input_size       
    )
    assert Path(DEFAULT_MODEL).with_suffix('.onnx').exists()

def test_export_with_custom_dynamic_axes(exporter):
    input_size = (640, 640)
    exporter.export(
        export_format='onnx', 
        input_size=input_size
    )
    assert Path(DEFAULT_MODEL).with_suffix('.onnx').exists()

def test_export_with_simplification(exporter):
    input_size = (640, 640)

    exporter.export(
        export_format='onnx', 
        input_size=input_size
    )
    assert Path(DEFAULT_MODEL).with_suffix('.onnx').exists()

def test_export_invalid_format(exporter):
    input_size = (640, 640)
    default_dynamic_axes = {'images': {0: 'N'}, 'orig_target_sizes': {0: 'N'}}
    
    with pytest.raises(ValueError, match="Export format is missing!"):
        exporter.export(
            export_format=None, 
            input_size=input_size            
        )

def test_openvino_export(exporter, tmp_path):
    input_size = (640, 640)

    exporter.export(
        export_format='openvino',
        input_size=input_size
    )
    assert Path(DEFAULT_MODEL).with_suffix('.xml').exists()

def test_openvino_export_fp16(exporter, tmp_path):
    input_size = (640, 640)
    exporter.export(
        export_format='openvino',
        input_size=input_size,
        fp16=True,
    )
    assert Path(DEFAULT_MODEL).with_suffix('.xml').exists()
