use super::*;

fn planes() -> (std::vec::Vec<u8>, std::vec::Vec<u8>, std::vec::Vec<u8>) {
  // 16×8 frame, U/V are 8×4.
  (
    std::vec![0u8; 16 * 8],
    std::vec![128u8; 8 * 4],
    std::vec![128u8; 8 * 4],
  )
}

#[test]
fn try_new_accepts_valid_tight() {
  let (y, u, v) = planes();
  let f = Yuv420pFrame::try_new(&y, &u, &v, 16, 8, 16, 8, 8).expect("valid");
  assert_eq!(f.width(), 16);
  assert_eq!(f.height(), 8);
}

#[test]
fn try_new_accepts_valid_padded_strides() {
  // 16×8 frame, strides padded (32 for y, 16 for u/v).
  let y = std::vec![0u8; 32 * 8];
  let u = std::vec![128u8; 16 * 4];
  let v = std::vec![128u8; 16 * 4];
  let f = Yuv420pFrame::try_new(&y, &u, &v, 16, 8, 32, 16, 16).expect("valid");
  assert_eq!(f.y_stride(), 32);
}

#[test]
fn try_new_rejects_zero_dim() {
  let (y, u, v) = planes();
  let e = Yuv420pFrame::try_new(&y, &u, &v, 0, 8, 16, 8, 8).unwrap_err();
  assert!(matches!(e, Yuv420pFrameError::ZeroDimension { .. }));
}

#[test]
fn try_new_rejects_odd_width() {
  let (y, u, v) = planes();
  let e = Yuv420pFrame::try_new(&y, &u, &v, 15, 8, 16, 8, 8).unwrap_err();
  assert!(matches!(e, Yuv420pFrameError::OddWidth { width: 15 }));
}

#[test]
fn try_new_accepts_odd_height() {
  // 16x9 frame — chroma_height = ceil(9/2) = 5. Y plane 16*9 = 144
  // bytes, U/V plane 8*5 = 40 bytes each. Valid 4:2:0 frame;
  // height=9 must not be rejected just because it's odd.
  let y = std::vec![0u8; 16 * 9];
  let u = std::vec![128u8; 8 * 5];
  let v = std::vec![128u8; 8 * 5];
  let f = Yuv420pFrame::try_new(&y, &u, &v, 16, 9, 16, 8, 8).expect("odd height valid");
  assert_eq!(f.height(), 9);
}

#[test]
fn try_new_rejects_y_stride_under_width() {
  let y = std::vec![0u8; 16 * 8];
  let u = std::vec![128u8; 8 * 4];
  let v = std::vec![128u8; 8 * 4];
  let e = Yuv420pFrame::try_new(&y, &u, &v, 16, 8, 8, 8, 8).unwrap_err();
  assert!(matches!(e, Yuv420pFrameError::YStrideTooSmall { .. }));
}

#[test]
fn try_new_rejects_short_y_plane() {
  let y = std::vec![0u8; 10];
  let u = std::vec![128u8; 8 * 4];
  let v = std::vec![128u8; 8 * 4];
  let e = Yuv420pFrame::try_new(&y, &u, &v, 16, 8, 16, 8, 8).unwrap_err();
  assert!(matches!(e, Yuv420pFrameError::YPlaneTooShort { .. }));
}

#[test]
fn try_new_rejects_short_u_plane() {
  let y = std::vec![0u8; 16 * 8];
  let u = std::vec![128u8; 4];
  let v = std::vec![128u8; 8 * 4];
  let e = Yuv420pFrame::try_new(&y, &u, &v, 16, 8, 16, 8, 8).unwrap_err();
  assert!(matches!(e, Yuv420pFrameError::UPlaneTooShort { .. }));
}

#[test]
#[should_panic(expected = "invalid Yuv420pFrame")]
fn new_panics_on_invalid() {
  let y = std::vec![0u8; 10];
  let u = std::vec![128u8; 8 * 4];
  let v = std::vec![128u8; 8 * 4];
  let _ = Yuv420pFrame::new(&y, &u, &v, 16, 8, 16, 8, 8);
}

// ---- Yuv410pFrame -------------------------------------------------------
//
// 4:1:0 (Cinepak / Sorenson legacy): width must be a multiple of 4;
// height may be any non-zero value; chroma planes are quarter-width and
// chroma height is computed with div_ceil(4).

fn yuv410p_planes() -> (std::vec::Vec<u8>, std::vec::Vec<u8>, std::vec::Vec<u8>) {
  // 16×8 frame → U/V are 4×2.
  (
    std::vec![0u8; 16 * 8],
    std::vec![128u8; 4 * 2],
    std::vec![128u8; 4 * 2],
  )
}

// ---- Yuv411pFrame::try_new -------------------------------------------

fn planes_411(w: u32, h: u32) -> (std::vec::Vec<u8>, std::vec::Vec<u8>, std::vec::Vec<u8>) {
  let ws = w as usize;
  let hs = h as usize;
  // FFmpeg `AV_PIX_FMT_YUV411P`: chroma row width = `width.div_ceil(4)`.
  let cw = ws.div_ceil(4);
  (
    std::vec![0u8; ws * hs],
    std::vec![128u8; cw * hs],
    std::vec![128u8; cw * hs],
  )
}

#[test]
fn yuv410p_try_new_accepts_valid_tight() {
  let (y, u, v) = yuv410p_planes();
  let f = Yuv410pFrame::try_new(&y, &u, &v, 16, 8, 16, 4, 4).expect("valid");
  assert_eq!(f.width(), 16);
  assert_eq!(f.height(), 8);
  assert_eq!(f.u_stride(), 4);
}

#[test]
fn yuv411p_try_new_accepts_valid_tight() {
  let (y, u, v) = planes_411(16, 8);
  let f = Yuv411pFrame::try_new(&y, &u, &v, 16, 8, 16, 4, 4).expect("valid");
  assert_eq!(f.width(), 16);
  assert_eq!(f.height(), 8);
  assert_eq!(f.u_stride(), 4);
}

#[test]
fn yuv410p_try_new_accepts_valid_padded_strides() {
  // 16×8 frame, strides padded.
  let y = std::vec![0u8; 32 * 8];
  let u = std::vec![128u8; 8 * 2];
  let v = std::vec![128u8; 8 * 2];
  let f = Yuv410pFrame::try_new(&y, &u, &v, 16, 8, 32, 8, 8).expect("valid");
  assert_eq!(f.y_stride(), 32);
}

#[test]
fn yuv410p_try_new_rejects_zero_dim() {
  let (y, u, v) = yuv410p_planes();
  let e = Yuv410pFrame::try_new(&y, &u, &v, 0, 8, 16, 4, 4).unwrap_err();
  assert!(matches!(e, Yuv410pFrameError::ZeroDimension { .. }));
}

#[test]
fn yuv410p_try_new_rejects_width_not_multiple_of_4() {
  // Width = 14 → not a multiple of 4. Even widths that aren't 4-aligned
  // are still rejected (unlike 4:2:0 which only requires even width).
  let y = std::vec![0u8; 14 * 8];
  let u = std::vec![128u8; 4 * 2];
  let v = std::vec![128u8; 4 * 2];
  let e = Yuv410pFrame::try_new(&y, &u, &v, 14, 8, 14, 4, 4).unwrap_err();
  assert!(matches!(
    e,
    Yuv410pFrameError::WidthNotMultipleOf4 { width: 14 }
  ));
}

#[test]
fn yuv410p_try_new_accepts_height_not_multiple_of_4() {
  // Height = 6 → not a multiple of 4. The walker (`chroma_row =
  // y_row / 4`) reuses chroma row 1 for Y rows 4 and 5, so chroma
  // height must be `height.div_ceil(4) = 2`. Accepts the frame.
  let y = std::vec![0u8; 16 * 6];
  let u = std::vec![128u8; 4 * 2];
  let v = std::vec![128u8; 4 * 2];
  let f = Yuv410pFrame::try_new(&y, &u, &v, 16, 6, 16, 4, 4).expect("valid");
  assert_eq!(f.height(), 6);
}

#[test]
fn yuv410p_try_new_accepts_height_10_with_three_chroma_rows() {
  // Height = 10 → chroma rows = ceil(10 / 4) = 3 (covering Y rows
  // 0..4, 4..8, and 8..10).
  let y = std::vec![0u8; 16 * 10];
  let u = std::vec![128u8; 4 * 3];
  let v = std::vec![128u8; 4 * 3];
  let f = Yuv410pFrame::try_new(&y, &u, &v, 16, 10, 16, 4, 4).expect("valid");
  assert_eq!(f.height(), 10);
}

#[test]
fn yuv410p_try_new_rejects_short_chroma_for_partial_group() {
  // Height = 6 → chroma rows must be `div_ceil(6, 4) = 2`. A plane
  // sized for floor(6 / 4) = 1 chroma row is rejected.
  let y = std::vec![0u8; 16 * 6];
  let u = std::vec![128u8; 4]; // only 1 chroma row, need 2
  let v = std::vec![128u8; 4 * 2];
  let e = Yuv410pFrame::try_new(&y, &u, &v, 16, 6, 16, 4, 4).unwrap_err();
  assert!(matches!(e, Yuv410pFrameError::UPlaneTooShort { .. }));
}

#[test]
fn yuv410p_try_new_rejects_y_stride_under_width() {
  let (y, u, v) = yuv410p_planes();
  let e = Yuv410pFrame::try_new(&y, &u, &v, 16, 8, 8, 4, 4).unwrap_err();
  assert!(matches!(e, Yuv410pFrameError::YStrideTooSmall { .. }));
}

#[test]
fn yuv410p_try_new_rejects_u_stride_under_chroma_width() {
  let (y, u, v) = yuv410p_planes();
  let e = Yuv410pFrame::try_new(&y, &u, &v, 16, 8, 16, 2, 4).unwrap_err();
  assert!(matches!(e, Yuv410pFrameError::UStrideTooSmall { .. }));
}

#[test]
fn yuv410p_try_new_rejects_short_y_plane() {
  let y = std::vec![0u8; 10];
  let u = std::vec![128u8; 4 * 2];
  let v = std::vec![128u8; 4 * 2];
  let e = Yuv410pFrame::try_new(&y, &u, &v, 16, 8, 16, 4, 4).unwrap_err();
  assert!(matches!(e, Yuv410pFrameError::YPlaneTooShort { .. }));
}

#[test]
fn yuv410p_try_new_rejects_short_u_plane() {
  let y = std::vec![0u8; 16 * 8];
  let u = std::vec![128u8; 2];
  let v = std::vec![128u8; 4 * 2];
  let e = Yuv410pFrame::try_new(&y, &u, &v, 16, 8, 16, 4, 4).unwrap_err();
  assert!(matches!(e, Yuv410pFrameError::UPlaneTooShort { .. }));
}

#[test]
fn yuv410p_try_new_rejects_short_v_plane() {
  let y = std::vec![0u8; 16 * 8];
  let u = std::vec![128u8; 4 * 2];
  let v = std::vec![128u8; 2];
  let e = Yuv410pFrame::try_new(&y, &u, &v, 16, 8, 16, 4, 4).unwrap_err();
  assert!(matches!(e, Yuv410pFrameError::VPlaneTooShort { .. }));
}

#[test]
#[should_panic(expected = "invalid Yuv410pFrame")]
fn yuv410p_new_panics_on_invalid() {
  let y = std::vec![0u8; 10];
  let u = std::vec![128u8; 4 * 2];
  let v = std::vec![128u8; 4 * 2];
  let _ = Yuv410pFrame::new(&y, &u, &v, 16, 8, 16, 4, 4);
}

#[cfg(target_pointer_width = "32")]
#[test]
fn yuv410p_try_new_rejects_y_geometry_overflow() {
  let big: u32 = 0x1_0000;
  let y: [u8; 0] = [];
  let u: [u8; 0] = [];
  let v: [u8; 0] = [];
  let e = Yuv410pFrame::try_new(&y, &u, &v, big, big, big, big / 4, big / 4).unwrap_err();
  assert!(matches!(e, Yuv410pFrameError::GeometryOverflow { .. }));
}

#[test]
fn yuv411p_try_new_rejects_zero_dim() {
  let (y, u, v) = planes_411(16, 8);
  let e = Yuv411pFrame::try_new(&y, &u, &v, 0, 8, 16, 4, 4).unwrap_err();
  assert!(matches!(e, Yuv411pFrameError::ZeroDimension { .. }));
}

#[test]
fn yuv411p_try_new_accepts_non_4_aligned_widths() {
  // FFmpeg `AV_PIX_FMT_YUV411P` defines chroma width via ceiling
  // right shift (`(w + 3) >> 2`). Construction must accept any
  // non-zero width and size chroma rows accordingly: 1→1, 2→1, 3→1,
  // 5→2, 6→2, 7→2, 17→5, 641→161.
  for (w, expected_cw) in [
    (1u32, 1u32),
    (2, 1),
    (3, 1),
    (5, 2),
    (6, 2),
    (7, 2),
    (9, 3),
    (17, 5),
    (641, 161),
  ] {
    let ws = w as usize;
    let cws = expected_cw as usize;
    let h: u32 = 4;
    let hs = h as usize;
    let y = std::vec![0u8; ws * hs];
    let u = std::vec![128u8; cws * hs];
    let v = std::vec![128u8; cws * hs];
    let f = Yuv411pFrame::try_new(&y, &u, &v, w, h, w, expected_cw, expected_cw)
      .unwrap_or_else(|e| panic!("width={w} (chroma={expected_cw}) should be valid: {e:?}"));
    assert_eq!(f.width(), w, "width={w}");
    assert_eq!(f.u_stride(), expected_cw, "width={w}");
    assert_eq!(f.v_stride(), expected_cw, "width={w}");
  }
}

#[test]
fn yuv411p_try_new_rejects_chroma_stride_below_div_ceil_4() {
  // Width=5 → chroma_width = 2; passing u_stride=1 must fail with
  // the new ceil-shift validator.
  let y = std::vec![0u8; 5 * 4];
  let u = std::vec![128u8; 2 * 4];
  let v = std::vec![128u8; 2 * 4];
  let e = Yuv411pFrame::try_new(&y, &u, &v, 5, 4, 5, 1, 2).unwrap_err();
  assert!(
    matches!(
      e,
      Yuv411pFrameError::UStrideTooSmall {
        chroma_width: 2,
        u_stride: 1
      }
    ),
    "{e:?}"
  );
}

#[test]
fn yuv411p_try_new_accepts_odd_height() {
  // 16x9 — chroma is full-height, so height=9 needs 9 rows of 4-byte
  // chroma each (no `div_ceil(2)` like 4:2:0).
  let y = std::vec![0u8; 16 * 9];
  let u = std::vec![128u8; 4 * 9];
  let v = std::vec![128u8; 4 * 9];
  let f = Yuv411pFrame::try_new(&y, &u, &v, 16, 9, 16, 4, 4).expect("odd height valid");
  assert_eq!(f.height(), 9);
}

#[test]
fn yuv411p_try_new_rejects_y_stride_under_width() {
  let (y, u, v) = planes_411(16, 8);
  let e = Yuv411pFrame::try_new(&y, &u, &v, 16, 8, 8, 4, 4).unwrap_err();
  assert!(matches!(e, Yuv411pFrameError::YStrideTooSmall { .. }));
}

#[test]
fn yuv411p_try_new_rejects_u_stride_under_chroma_width() {
  let (y, u, v) = planes_411(16, 8);
  // chroma_width = 4; passing u_stride = 3 must fail.
  let e = Yuv411pFrame::try_new(&y, &u, &v, 16, 8, 16, 3, 4).unwrap_err();
  assert!(matches!(e, Yuv411pFrameError::UStrideTooSmall { .. }));
}

#[test]
fn yuv411p_try_new_rejects_short_y_plane() {
  let y = std::vec![0u8; 10];
  let u = std::vec![128u8; 4 * 8];
  let v = std::vec![128u8; 4 * 8];
  let e = Yuv411pFrame::try_new(&y, &u, &v, 16, 8, 16, 4, 4).unwrap_err();
  assert!(matches!(e, Yuv411pFrameError::YPlaneTooShort { .. }));
}

#[test]
fn yuv411p_try_new_rejects_short_u_plane() {
  let y = std::vec![0u8; 16 * 8];
  let u = std::vec![128u8; 4];
  let v = std::vec![128u8; 4 * 8];
  let e = Yuv411pFrame::try_new(&y, &u, &v, 16, 8, 16, 4, 4).unwrap_err();
  assert!(matches!(e, Yuv411pFrameError::UPlaneTooShort { .. }));
}

#[test]
#[should_panic(expected = "invalid Yuv411pFrame")]
fn yuv411p_new_panics_on_invalid() {
  let y = std::vec![0u8; 10];
  let u = std::vec![128u8; 4 * 8];
  let v = std::vec![128u8; 4 * 8];
  let _ = Yuv411pFrame::new(&y, &u, &v, 16, 8, 16, 4, 4);
}
