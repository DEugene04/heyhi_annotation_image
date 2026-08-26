"""
Tests for turning a Mathpix response into the IR.

The response below is shaped like a real Mathpix v3/text reply (the fields the
docs define), with a printed line, two handwritten maths lines, a handwritten
text line, and a drawn figure — so we can check each is handled correctly
without calling the live service. Every line of writing is kept (telling answer
from question is the evaluator's job); only the drawn figure is set aside.
"""

from contracts.schema import SegmentType
from ocr.ir_builder import area_weighted_confidence, is_readable, parse_ir

RESPONSE = {
    "line_data": [
        {  # printed line — kept, not dropped
            "id": "l0", "type": "text", "is_printed": True, "is_handwritten": False,
            "text": "Solve for x", "confidence": 0.99,
            "cnt": [[10, 10], [200, 10], [200, 40], [10, 40]],
        },
        {  # handwritten maths
            "id": "l1", "type": "math", "is_handwritten": True,
            "text": "\\( 2x + 4 = 10 \\)", "confidence": 0.97,
            "cnt": [[10, 50], [300, 50], [300, 90], [10, 90]],
        },
        {  # handwritten words
            "id": "l2", "type": "text", "is_handwritten": True,
            "text": "so we solve", "confidence": 0.90,
            "cnt": [[10, 100], [250, 100], [250, 140], [10, 140]],
        },
        {  # a drawn figure — routed to the diagram track
            "id": "l3", "type": "diagram", "is_handwritten": True,
            "text": "", "confidence": 0.80,
            "cnt": [[10, 150], [400, 150], [400, 400], [10, 400]],
        },
        {  # handwritten maths again
            "id": "l4", "type": "math", "is_handwritten": True,
            "text": "\\( x = 3 \\)", "confidence": 0.95,
            "cnt": [[10, 420], [200, 420], [200, 460], [10, 460]],
        },
    ]
}


def _ir():
    return parse_ir(RESPONSE, image_width=500, image_height=500)


def test_all_text_lines_become_segments():
    ir = _ir()
    assert [s.id for s in ir.segments] == ["l0", "l1", "l2", "l4"]  # figure aside


def test_diagram_is_routed_to_its_own_track():
    ir = _ir()
    assert len(ir.diagrams) == 1
    assert ir.diagrams[0].type == SegmentType.DIAGRAM


def test_char_ranges_index_into_flat_text():
    ir = _ir()
    for segment in ir.segments:
        assert ir.flat_text[segment.char_start : segment.char_end] == segment.text


def test_flat_text_joins_lines_with_breaks():
    ir = _ir()
    assert ir.flat_text == "Solve for x\n\\( 2x + 4 = 10 \\)\nso we solve\n\\( x = 3 \\)"


def test_line_numbers_keep_page_gaps():
    # The diagram line leaves a gap, so the two maths lines around the figure are
    # not adjacent — which is what keeps them as separate regions.
    ir = _ir()
    assert [s.line for s in ir.segments] == [0, 1, 2, 4]


def test_latex_is_extracted_for_maths_only():
    ir = _ir()
    by_id = {s.id: s for s in ir.segments}
    assert by_id["l1"].latex == "2x + 4 = 10"
    assert by_id["l4"].latex == "x = 3"
    assert by_id["l2"].latex is None


def test_quad_is_four_corners_from_the_contour():
    ir = _ir()
    quad = ir.segments[1].quad  # the first maths line
    assert len(quad) == 4
    assert (quad[0].x, quad[0].y) == (10, 50)  # top-left
    assert (quad[2].x, quad[2].y) == (300, 90)  # bottom-right


def test_confidence_gate():
    ir = _ir()
    assert area_weighted_confidence(ir) > 0.9
    assert is_readable(ir) is True
