use crate::game::Vertex;

// 7-segment bits: a=1 b=2 c=4 d=8 e=16 f=32 g=64
fn mask(c: char) -> u8 {
    match c {
        '0' => 0b0111111,
        '1' => 0b0000110,
        '2' => 0b1011011,
        '3' => 0b1001111,
        '4' => 0b1100110,
        '5' => 0b1101101,
        '6' => 0b1111101,
        '7' => 0b0000111,
        '8' => 0b1111111,
        '9' => 0b1101111,
        'F' => 0b1110001,
        'P' => 0b1110011,
        'S' => 0b1101101,
        'A' => 0b1110111,
        'I' => 0b0000110,
        _ => 0,
    }
}

fn rect(out: &mut Vec<Vertex>, x0: f32, y0: f32, x1: f32, y1: f32, z: f32, c: [f32; 3]) {
    for &(x, y) in &[(x0, y0), (x1, y0), (x1, y1), (x0, y0), (x1, y1), (x0, y1)] {
        out.push(Vertex { pos: [x, y, z], color: c, kind: 1.0 });
    }
}

/// "FPS nn" in the top-left corner, in clip space (identity view-projection).
pub fn build_fps(out: &mut Vec<Vertex>, fps: u32, aspect: f32) {
    let text = format!("FPS {}", fps.min(999));
    let h = 0.045;
    let w = h * 0.55 / aspect;
    let t = h * 0.18; // horizontal bar thickness (y)
    let tx = t / aspect; // vertical bar thickness (x)
    let gap = w * 0.45;
    let x0 = -0.975;
    let yt = 0.965;
    let yb = yt - h;
    let ym = yt - h * 0.5;

    let chars = text.chars().count() as f32;
    let total = chars * (w + gap) - gap;
    let pad_x = 0.008 / aspect * 2.0;
    let pad_y = 0.008;
    rect(
        out,
        x0 - pad_x,
        yb - pad_y,
        x0 + total + pad_x,
        yt + pad_y,
        0.001,
        [0.04, 0.05, 0.08],
    );

    let col = [0.25, 1.0, 0.35];
    let mut cx = x0;
    for ch in text.chars() {
        let m = mask(ch);
        if m & 1 != 0 { rect(out, cx, yt - t, cx + w, yt, 0.0, col); }
        if m & 2 != 0 { rect(out, cx + w - tx, ym, cx + w, yt, 0.0, col); }
        if m & 4 != 0 { rect(out, cx + w - tx, yb, cx + w, ym, 0.0, col); }
        if m & 8 != 0 { rect(out, cx, yb, cx + w, yb + t, 0.0, col); }
        if m & 16 != 0 { rect(out, cx, yb, cx + tx, ym, 0.0, col); }
        if m & 32 != 0 { rect(out, cx, ym, cx + tx, yt, 0.0, col); }
        if m & 64 != 0 { rect(out, cx, ym - t * 0.5, cx + w, ym + t * 0.5, 0.0, col); }
        cx += w + gap;
    }
}

/// Clip-space rectangle (x0, y0, x1, y1) of the small "AI" toggle button under the FPS bar.
pub fn ai_button_rect(aspect: f32) -> (f32, f32, f32, f32) {
    let x0 = -0.975;
    let y1 = 0.885;
    (x0, y1 - 0.085, x0 + 0.21 / aspect, y1)
}

/// Small "AI" button; green when the AI rider is on.
pub fn build_ai_button(out: &mut Vec<Vertex>, aspect: f32, on: bool) {
    let (x0, y0, x1, y1) = ai_button_rect(aspect);
    let (bg, fg, edge) = if on {
        ([0.08, 0.50, 0.20], [1.0, 1.0, 1.0], [0.35, 1.0, 0.5])
    } else {
        ([0.04, 0.05, 0.08], [0.55, 0.62, 0.7], [0.25, 0.3, 0.38])
    };
    let e = 0.006;
    let ex = e / aspect;
    rect(out, x0, y0, x1, y1, 0.001, edge);
    rect(out, x0 + ex, y0 + e, x1 - ex, y1 - e, 0.0005, bg);
    let h = 0.048;
    let w = h * 0.55 / aspect;
    let t = h * 0.18;
    let tx = t / aspect;
    let gap = w * 0.5;
    let total = 2.0 * w + gap;
    let mut cx = (x0 + x1) * 0.5 - total * 0.5;
    let yt = (y0 + y1) * 0.5 + h * 0.5;
    let yb = yt - h;
    let ym = yt - h * 0.5;
    for ch in ['A', 'I'] {
        let m = mask(ch);
        if m & 1 != 0 { rect(out, cx, yt - t, cx + w, yt, 0.0, fg); }
        if m & 2 != 0 { rect(out, cx + w - tx, ym, cx + w, yt, 0.0, fg); }
        if m & 4 != 0 { rect(out, cx + w - tx, yb, cx + w, ym, 0.0, fg); }
        if m & 8 != 0 { rect(out, cx, yb, cx + w, yb + t, 0.0, fg); }
        if m & 16 != 0 { rect(out, cx, yb, cx + tx, ym, 0.0, fg); }
        if m & 32 != 0 { rect(out, cx, ym, cx + tx, yt, 0.0, fg); }
        if m & 64 != 0 { rect(out, cx, ym - t * 0.5, cx + w, ym + t * 0.5, 0.0, fg); }
        cx += w + gap;
    }
}
