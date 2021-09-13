#[macro_use]
extern crate rocket;
extern crate serde;

use rocket::http::ContentType;
use rocket::response::content;
use rocket::serde::{json::Json, Serialize};
use rocket::State;
use std::collections::HashSet;
use std::net::IpAddr;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::thread;
// use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::Duration;

#[derive(Serialize, Copy, Clone, Hash, Debug, PartialEq, Eq)]
pub struct RvxColor {
    pub rgb: u32,
    a: BwF64,
}

impl RvxColor {
    pub const WHITE: RvxColor = RvxColor::rgb(0xff, 0xff, 0xff);
    pub const BLACK: RvxColor = RvxColor::rgb(0x00, 0x00, 0x00);
    pub const RED: RvxColor = RvxColor::rgb(0xff, 0x00, 0x00);
    pub const GREEN: RvxColor = RvxColor::rgb(0x00, 0xff, 0x00);
    pub const BLUE: RvxColor = RvxColor::rgb(0x00, 0x00, 0xff);
    pub const CYAN: RvxColor = RvxColor::rgb(0x00, 0xff, 0xff);
    pub const MAGENTA: RvxColor = RvxColor::rgb(0xff, 0x00, 0xff);
    pub const YELLOW: RvxColor = RvxColor::rgb(0xff, 0xff, 0x00);
    pub const LIGHT_GRAY: RvxColor = RvxColor::rgb(0xd3, 0xd3, 0xd3);
    pub const GRAY: RvxColor = RvxColor::rgb(0x7f, 0x7f, 0x7f);
    pub const DARK_GRAY: RvxColor = RvxColor::rgb(0x55, 0x55, 0x55);
    pub const PURPLE: RvxColor = RvxColor::rgb(0x63, 0x26, 0xcc);
    pub const MAROON: RvxColor = RvxColor::rgb(0x7f, 0x00, 0x00);
    pub const FOREST: RvxColor = RvxColor::rgb(0x00, 0x7f, 0x00);
    pub const NAVY: RvxColor = RvxColor::rgb(0x00, 0x00, 0x7f);
    pub const OLIVE: RvxColor = RvxColor::rgb(0x00, 0x7f, 0x7f);
    pub const PLUM: RvxColor = RvxColor::rgb(0x7f, 0x00, 0x7f);
    pub const TEAL: RvxColor = RvxColor::rgb(0x7f, 0x7f, 0x00);
    pub const ORANGE: RvxColor = RvxColor::rgb(0xff, 0x7f, 0x00);
    pub const NEON_GREEN: RvxColor = RvxColor::rgb(0x7f, 0xff, 0x00);
    pub const PINK: RvxColor = RvxColor::rgb(0xff, 0xc0, 0xcb);

    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self::rgba(r, g, b, 1.0)
    }

    pub const fn rgba(r: u8, g: u8, b: u8, a: f64) -> Self {
        let rgb = (r as u32) << 16 | (g as u32) << 8 | b as u32;
        Self { rgb, a: BwF64(a) }
    }

    pub fn scale_rgb(&self, factor: f32) -> Self {
        Self::rgba(
            (self.r() as f32 * factor).min(255.0).max(0.0) as u8,
            (self.g() as f32 * factor).min(255.0).max(0.0) as u8,
            (self.b() as f32 * factor).min(255.0).max(0.0) as u8,
            self.a(),
        )
    }

    #[allow(unused)]
    pub fn a(&self) -> f64 {
        self.a.0
    }

    pub fn r(&self) -> u8 {
        ((self.rgb >> 16) & 0xff) as u8
    }

    pub fn g(&self) -> u8 {
        ((self.rgb >> 8) & 0xff) as u8
    }

    pub fn b(&self) -> u8 {
        (self.rgb & 0xff) as u8
    }

    pub fn set_a(mut self, a: f64) -> Self {
        self.a = a.into();
        self
    }
}

// A bit-wise comparison f64
#[repr(transparent)]
#[derive(Serialize, Clone, Copy, Debug, Default)]
struct BwF64(f64);

impl PartialEq for BwF64 {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}

impl Eq for BwF64 {}

impl Hash for BwF64 {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

impl From<f64> for BwF64 {
    fn from(f: f64) -> Self {
        BwF64(f)
    }
}

impl From<&f64> for BwF64 {
    fn from(f: &f64) -> Self {
        BwF64(*f)
    }
}

#[derive(Serialize, Clone, Hash, Debug, PartialEq, Eq)]
enum ShapeType {
    TransformedOnly,
    Deleted,
    Circle,
    Square,
    Line {
        points: [BwF64; 4],
        line_width: BwF64,
    },
    Lines {
        points: Vec<BwF64>,
        line_width: BwF64,
    },
    Polygon {
        points: Vec<BwF64>,
    },
    Grid {
        line_width: BwF64,
    },
    Bitmap8 {
        image: Vec<u8>,
        width: usize,
        palette: Option<Vec<RvxColor>>,
    },
    Text {
        text: String,
        font_family: String,
        font_size: BwF64,
    },
    Array {
        shape: Box<Shape>,
        points: Vec<BwF64>,
    },
}

impl Default for ShapeType {
    fn default() -> Self {
        Self::TransformedOnly
    }
}

#[derive(Serialize, Clone, Copy, Hash, Debug, PartialEq, Eq)]
struct ShapeOuter {
    idx: usize,
    x: BwF64,
    y: BwF64,
    rot: BwF64,
    scale: (BwF64, BwF64),
    color: u32,
    alpha: BwF64,
}

impl Default for ShapeOuter {
    fn default() -> Self {
        Self {
            idx: 0,
            x: 0.0.into(),
            y: 0.0.into(),
            rot: 0.0.into(),
            scale: (1.0.into(), 1.0.into()),
            color: RvxColor::WHITE.rgb,
            alpha: 1.0.into(),
        }
    }
}

#[derive(Serialize, Clone, Hash, Default, Debug, PartialEq, Eq)]
pub struct Shape {
    outer: ShapeOuter,
    inner: ShapeType,
}

impl Shape {
    #[must_use]
    pub fn scale(mut self, scale: f64) -> Shape {
        self.outer.scale.0 = ((self.outer.scale.0).0 * scale).into();
        self.outer.scale.1 = ((self.outer.scale.1).0 * scale).into();
        self
    }

    #[must_use]
    pub fn scale_xy(mut self, scale_xy: &[f64]) -> Shape {
        assert_eq!(scale_xy.len(), 2);
        self.outer.scale.0 = ((self.outer.scale.0).0 * scale_xy[0]).into();
        self.outer.scale.1 = ((self.outer.scale.1).0 * scale_xy[1]).into();
        self
    }

    #[must_use]
    pub fn rot(mut self, rot: f64) -> Shape {
        self.outer.rot = (self.outer.rot.0 + rot).into();
        self
    }

    #[must_use]
    pub fn translate(mut self, xy: &[f64]) -> Shape {
        assert_eq!(xy.len(), 2);
        self.outer.x = ((self.outer.x).0 + xy[0]).into();
        self.outer.y = ((self.outer.y).0 + xy[1]).into();
        self
    }

    #[must_use]
    pub fn xyt(mut self, xyt: &[f64]) -> Shape {
        assert_eq!(xyt.len(), 3);
        self.outer.x = ((self.outer.x).0 + xyt[0]).into();
        self.outer.y = ((self.outer.y).0 + xyt[1]).into();
        self.outer.rot = (self.outer.rot.0 + xyt[2]).into();
        self
    }

    #[must_use]
    pub fn color(mut self, color: RvxColor) -> Shape {
        self.outer.color = color.rgb;
        self.outer.alpha = color.a;
        self
    }

    #[must_use]
    pub fn rgb(mut self, rgb: u32) -> Shape {
        self.outer.color = rgb;
        self
    }

    #[must_use]
    pub fn alpha(mut self, alpha: f64) -> Shape {
        self.outer.alpha = alpha.into();
        self
    }
}

#[derive(Serialize, Clone)]
struct HashedShape {
    hash: (u32, u32),
    shape: Shape,
}

fn examine_changed(
    shapes: &mut Vec<HashedShape>,
    last_hashes: &Vec<(u32, u32)>,
) -> Vec<HashedShape> {
    let mut changes = Vec::new();

    let min_len = shapes.len().min(last_hashes.len());
    for i in 0..min_len {
        let shape = &shapes[i];
        let last_hash = last_hashes[i];

        // nothing at all changing, so don't even send it
        if shape.hash == last_hash {
            continue;
        }

        // inner shape didn't change, so we just change the our transform
        if shape.hash.1 == last_hash.1 {
            let changed_shape = HashedShape {
                hash: shape.hash,
                shape: Shape {
                    outer: shape.shape.outer,
                    inner: ShapeType::TransformedOnly,
                },
            };
            changes.push(changed_shape);
        } else {
            changes.push(shape.clone());
        }
    }
    for shape in shapes.iter().skip(min_len) {
        changes.push(shape.clone());
    }
    for _ in min_len..last_hashes.len() {
        changes.push(HashedShape {
            hash: (0, 0),
            shape: Shape {
                outer: ShapeOuter {
                    idx: min_len,
                    ..Default::default()
                },
                inner: ShapeType::Deleted,
            },
        });
    }
    changes
}

#[derive(Serialize)]
struct Packet {
    title: String,
    shapes: Vec<HashedShape>,
    user_zoom: Option<f64>,
    global_rot: f64,
}

#[post("/graphics_update", format = "json", data = "<hashes>")]
fn handle_graphics_update(data: &State<Rvx>, hashes: Json<Vec<(u32, u32)>>) -> Json<Packet> {
    let mut active = data.active.lock().unwrap();
    let changes = examine_changed(&mut active, &*hashes);
    Json(Packet {
        title: data.title.clone(),
        shapes: changes,
        user_zoom: *data.user_zoom.lock().unwrap(),
        global_rot: *data.global_rot.lock().unwrap(),
    })
}

const INDEX_HTML: &'static str = include_str!("../index.html");
const PIXI_JS: &'static str = include_str!("../pixi.min.js");
const PIXI_JS_MAP: &'static str = include_str!("../pixi.min.js.map");

#[get("/")]
fn handle_index_html(data: &State<Rvx>) -> content::Html<String> {
    let mut index_html = INDEX_HTML.to_owned();
    let offset = index_html.find("{{}}");
    if let Some(offset) = offset {
        index_html.replace_range(offset..offset + 4, &data.title);
    }

    content::Html(index_html)
}

#[get("/pixi.min.js")]
fn handle_pixi_js() -> content::JavaScript<&'static str> {
    content::JavaScript(PIXI_JS)
}

#[get("/pixi.min.js.map")]
fn handle_pixi_js_map() -> content::Custom<&'static str> {
    content::Custom(ContentType::Binary, PIXI_JS_MAP)
}

// #[actix_rt::main]
// async fn run_server(r: Rvx, addr_port: String, stop_rx: mpsc::Receiver<()>) -> std::io::Result<()> {
//     let server = HttpServer::new(move || {
//         App::new()
//             .wrap(middleware::Compress::default())
//             .data(r.clone())
//             .app_data(web::JsonConfig::default().limit(128 * 1024))
//             .route("/", web::get().to(handle_index_html))
//             .route("/pixi.min.js", web::get().to(handle_pixi_js))
//             .route("/pixi.min.js.map", web::get().to(handle_pixi_js_map))
//             .route("/graphics_update", web::post().to(handle_graphics_update))
//     })
//     .disable_signals()
//     .bind(addr_port)?
//     .run();
//
//     let thread_server = server.clone();
//     thread::spawn(move || {
//         stop_rx.recv().unwrap();
//         executor::block_on(thread_server.stop(false));
//     });
//
//     server.await
// }

fn run_server(r: Rvx, addr: [u8; 4], port: u16, running: Arc<AtomicBool>) {
    thread::spawn(move || {
        let mut config = rocket::Config::debug_default();
        config.log_level = rocket::log::LogLevel::Critical;
        config.address = IpAddr::from(addr);
        config.port = port;
        config.shutdown = rocket::config::Shutdown {
            ctrlc: false,
            signals: HashSet::new(),
            ..Default::default()
        };

        let rocket_future = rocket::custom(config.clone())
            .manage(r)
            .mount(
                "/",
                routes![
                    handle_index_html,
                    handle_pixi_js,
                    handle_pixi_js_map,
                    handle_graphics_update,
                ],
            )
            .launch();

        // borrowed from unstable rocket::async_main
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(config.workers)
            // NOTE: graceful shutdown depends on the "rocket-worker" prefix.
            .thread_name("rocket-worker-thread")
            .enable_all()
            .build()
            .expect("create tokio runtime")
            .block_on(rocket_future)
            .unwrap();
    });

    while running.load(Ordering::SeqCst) {
        thread::sleep(Duration::from_micros(100));
    }
}

#[derive(Clone)]
pub struct Rvx {
    title: String,
    alpha_modifier: f64,
    x_scale_modifier: f64,
    y_scale_modifier: f64,
    x_modifier: f64,
    y_modifier: f64,
    global_rot: Arc<Mutex<f64>>,
    working: Vec<Shape>,
    last: Vec<HashedShape>,
    active: Arc<Mutex<Vec<HashedShape>>>,
    user_zoom: Arc<Mutex<Option<f64>>>,
    thread: Option<Arc<thread::JoinHandle<()>>>,
    running: Arc<AtomicBool>,
}

impl Rvx {
    pub fn new(title: &str, addr: [u8; 4], port: u16) -> Self {
        let running = Arc::new(AtomicBool::new(true));

        let mut r = Rvx {
            title: title.to_owned(),
            alpha_modifier: 1.0,
            x_scale_modifier: 1.0,
            y_scale_modifier: 1.0,
            x_modifier: 0.0,
            y_modifier: 0.0,
            global_rot: Arc::new(Mutex::new(0.0f64)),
            working: Vec::new(),
            last: Vec::new(),
            active: Arc::new(Mutex::new(Vec::new())),
            user_zoom: Arc::new(Mutex::new(None)),
            thread: None,
            running: running.clone(),
        };
        let thread_r = r.clone();
        let thread_addr = addr.to_owned();
        r.thread = Some(Arc::new(thread::spawn(move || {
            run_server(thread_r, thread_addr, port, running);
        })));
        r
    }

    // use None to let user choose themself
    pub fn set_user_zoom(&mut self, zoom: Option<f64>) {
        *self.user_zoom.lock().unwrap() = zoom;
    }

    pub fn clear(&mut self) {
        self.working.clear();
        self.alpha_modifier = 1.0;
        self.x_scale_modifier = 1.0;
        self.y_scale_modifier = 1.0;
        self.x_modifier = 0.0;
        self.y_modifier = 0.0;
    }

    fn convert_vec(v: &[f64]) -> Vec<BwF64> {
        let mut new_v = Vec::with_capacity(v.len());
        for a in v.iter() {
            new_v.push(a.into());
        }
        new_v
    }

    pub fn set_alpha_modifier(&mut self, alpha: f64) {
        self.alpha_modifier = alpha;
    }

    pub fn set_translate_modifier(&mut self, x: f64, y: f64) {
        self.x_modifier = x;
        self.y_modifier = y;
    }

    pub fn set_scale_modifier(&mut self, scale_x: f64, scale_y: f64) {
        self.x_scale_modifier = scale_x;
        self.y_scale_modifier = scale_y;
    }

    pub fn set_global_rot(&mut self, rot: f64) {
        *self.global_rot.lock().unwrap() = rot;
    }

    // fn shape_outer_defaults(x: f64, y: f64, rot: f64, color: RvxColor) -> ShapeOuter {
    //     ShapeOuter {
    //         idx: 0,
    //         x: x.into(), y: y.into(), rot: rot.into(),
    //         color: color.rgb, alpha: color.a,
    //         ..Default::default()
    //     }
    // }

    pub fn circle() -> Shape {
        Shape {
            outer: Default::default(),
            inner: ShapeType::Circle,
        }
    }

    pub fn square() -> Shape {
        Shape {
            outer: Default::default(),
            inner: ShapeType::Square,
        }
    }

    pub fn robot() -> Shape {
        let length = 1.0;
        let width = 0.45;
        let robot_poly = vec![
            -length / 2.0,
            width / 2.0,
            -length / 2.0,
            -width / 2.0,
            length / 2.0,
            0.0,
        ];
        let points = Rvx::convert_vec(&robot_poly);
        // let robot_poly = unsafe { std::mem::transmute::<_, Vec<BwF64>>(robot_poly) };

        Shape {
            outer: Default::default(),
            inner: ShapeType::Polygon { points },
        }
    }

    pub fn line(points: [f64; 4], line_width: f64) -> Shape {
        let points = [
            points[0].into(),
            points[1].into(),
            points[2].into(),
            points[3].into(),
        ];

        Shape {
            outer: Default::default(),
            inner: ShapeType::Line {
                points,
                line_width: line_width.into(),
            },
        }
    }

    pub fn lines(points: &[f64], line_width: f64) -> Shape {
        assert_eq!(points.len() % 2, 0);

        let points = Rvx::convert_vec(&points);

        Shape {
            outer: Default::default(),
            inner: ShapeType::Lines {
                points,
                line_width: line_width.into(),
            },
        }
    }

    pub fn polygon(points: &[f64]) -> Shape {
        assert_eq!(points.len() % 2, 0);

        let points = Rvx::convert_vec(&points);

        Shape {
            outer: Default::default(),
            inner: ShapeType::Polygon { points },
        }
    }

    pub fn grid(line_width: f64) -> Shape {
        Shape {
            outer: Default::default(),
            inner: ShapeType::Grid {
                line_width: line_width.into(),
            },
        }
    }

    pub fn bitmap8(image: Vec<u8>, width: usize, palette: Option<Vec<RvxColor>>) -> Shape {
        Shape {
            outer: Default::default(),
            inner: ShapeType::Bitmap8 {
                image,
                width,
                palette,
            },
        }
    }

    pub fn text(text: &str, font_family: &str, font_size: f64) -> Shape {
        if text.len() > 4096 {
            eprintln!(
                "RVX Warning: text element starting with \"{}\" has length {}",
                &text[0..128],
                text.len()
            );
        }
        Shape {
            outer: Default::default(),
            inner: ShapeType::Text {
                text: text.to_owned(),
                font_family: font_family.to_owned(),
                font_size: font_size.into(),
            },
        }
    }

    pub fn array(shape: Shape, points: &[f64]) -> Shape {
        let points = Rvx::convert_vec(&points);
        let color = shape.outer.color;
        let alpha = shape.outer.alpha.0;
        Shape {
            outer: Default::default(),
            inner: ShapeType::Array {
                shape: Box::new(shape),
                points,
            },
        }
        .rgb(color)
        .alpha(alpha)
    }

    pub fn draw(&mut self, mut shape: Shape) {
        shape.outer.idx = self.working.len();
        shape.outer.x = BwF64(shape.outer.x.0 + self.x_modifier);
        shape.outer.y = BwF64(shape.outer.y.0 + self.y_modifier);
        shape.outer.scale.0 = BwF64((shape.outer.scale.0).0 * self.x_scale_modifier);
        shape.outer.scale.1 = BwF64((shape.outer.scale.1).0 * self.y_scale_modifier);
        shape.outer.alpha = BwF64(shape.outer.alpha.0 * self.alpha_modifier);

        if let ShapeType::Bitmap8 { palette, .. } = &mut shape.inner {
            if let Some(palette) = palette.as_mut() {
                for color in palette.iter_mut() {
                    color.a = BwF64(color.a.0 * self.alpha_modifier);
                }
            }
        }

        self.working.push(shape);
    }

    pub fn draw_all<T>(&mut self, shapes: T)
    where
        T: IntoIterator<Item = Shape>,
    {
        for shape in shapes.into_iter() {
            self.draw(shape);
        }
    }

    pub fn commit_changes(&mut self) {
        let mut new_active = Vec::with_capacity(self.working.len());
        for shape in self.working.iter().cloned() {
            let mut hasher = DefaultHasher::new();
            shape.outer.hash(&mut hasher);
            let outer_hash = hasher.finish();
            let outer_hash = (outer_hash >> 32) as u32 ^ (outer_hash & 0xffffffff) as u32;

            let mut hasher = DefaultHasher::new();
            shape.inner.hash(&mut hasher);
            // color also requires redrawing the basic element
            shape.outer.color.hash(&mut hasher);
            let inner_hash = hasher.finish();
            let inner_hash = (inner_hash >> 32) as u32 ^ (inner_hash & 0xffffffff) as u32;

            let hash = (outer_hash, inner_hash);
            new_active.push(HashedShape { hash, shape });
        }

        // check serialized size of new_active to see if maybe there is a user error
        // and we have a set that is just too large!
        let serialized_size = serde_json::to_vec(&new_active).unwrap().len();
        let kb_size = serialized_size as f32 / 1024.0;
        if kb_size >= 1000.0 {
            eprintln!("Serialized (JSON) size: {:.2}KB", kb_size);
        }

        self.last = new_active.clone();
        *self.active.lock().unwrap() = new_active;
    }

    pub fn shapes(&self) -> &[Shape] {
        &self.working
    }
}

impl Drop for Rvx {
    fn drop(&mut self) {
        if let Some(thread) = self.thread.take() {
            if let Some(thread) = Arc::try_unwrap(thread).ok() {
                self.running.store(false, Ordering::SeqCst);
                thread.join().unwrap();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_static() {
        let mut r = Rvx::new("Rvx Test", [0, 0, 0, 0], 3000);

        let color = RvxColor::RED.set_a(0.9);

        r.draw(Rvx::grid(2.0));

        r.draw(
            Rvx::circle()
                .translate(&[-1.0, 2.0])
                .scale_xy(&[2.0, 1.0])
                .rot(-0.1)
                .color(color),
        );

        r.commit_changes();
        std::thread::sleep(std::time::Duration::from_secs(4));
        panic!();
    }
}
