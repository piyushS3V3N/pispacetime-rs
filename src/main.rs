use bevy::math::primitives::Sphere;
use bevy::mesh::Mesh3d;
use bevy::pbr::MeshMaterial3d;
use bevy::prelude::*;
use bevy_flycam::prelude::*;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::collections::VecDeque;
use std::f32::consts::TAU;

use bevy_egui::{egui, EguiContexts, EguiPlugin, EguiPrimaryContextPass};

// ================= CONSTANTS =================

// engine units per AU (smaller => system looks bigger)
const SCALE: f32 = 6.0;

// how far (in AU) the sheet extends visually
const FUNNEL_RADIUS_AU: f32 = 16.0;

// overall depth of the central well (bigger = deeper)
const FUNNEL_DEPTH: f32 = 6.0;

// "core radius" in AU: makes the center rounded instead of a sharp spike
const FUNNEL_SOFTEN_R_AU: f32 = 1.0;

// simulation speed: years of orbit per real second
const YEARS_PER_SECOND: f32 = 0.3;

// number of trail samples per body
const TRAIL_LEN: usize = 1500;

// ================= DATA TYPES =================

#[derive(Clone)]
struct OrbitBody {
    name: String,
    radius_au: f32,
    period_years: f32,
    angle: f32, // current orbital angle (radians)
    color: Color,
    size: f32, // visual radius in world units
}

#[derive(Resource)]
struct Orbits {
    bodies: Vec<OrbitBody>,
}

#[derive(Resource)]
struct Trails {
    points: Vec<VecDeque<Vec3>>, // world positions on the sheet
}

#[derive(Resource)]
struct RngRes(StdRng);

#[derive(Component)]
struct BodyIndex(usize);

// ----------- Tab state for the docs window -----------

#[derive(Clone, Copy, PartialEq, Eq)]
enum DocTab {
    Overview,
    Equations,
    RealPhysics,
}

#[derive(Resource)]
struct DocTabState {
    current: DocTab,
}

// ----------- Mini N-body schematic data -----------

#[derive(Clone)]
struct SchematicBody {
    mass: f32,
    pos: Vec2,
    vel: Vec2,
    color: egui::Color32,
    radius_px: f32,
}

#[derive(Resource)]
struct SchematicSystem {
    bodies: Vec<SchematicBody>,
}

// ================= UTILITIES =================

fn funnel_height(r_au: f32) -> f32 {
    // soften the center so r never truly reaches 0
    let r_soft = (r_au * r_au + FUNNEL_SOFTEN_R_AU * FUNNEL_SOFTEN_R_AU).sqrt();
    let phi = -1.0 / r_soft;

    // value of phi at the edge of the funnel, so we can shift the whole thing
    let edge_r = FUNNEL_RADIUS_AU + FUNNEL_SOFTEN_R_AU;
    let phi_edge = -1.0 / edge_r;

    (phi - phi_edge) * FUNNEL_DEPTH
}

// world coords on the funnel for a given (radius, angle)
fn orbit_position_on_funnel(radius_au: f32, angle: f32) -> Vec3 {
    let x_world = radius_au * angle.cos() * SCALE;
    let z_world = radius_au * angle.sin() * SCALE;
    let y_world = funnel_height(radius_au);
    Vec3::new(x_world, y_world, z_world)
}

// ================= SETUP MAIN SCENE =================

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.insert_resource(ClearColor(Color::srgb(0.02, 0.02, 0.06)));

    // ---- Define cartoon solar system ----
    let mut bodies = Vec::new();

    bodies.push(OrbitBody {
        name: "Sun".into(),
        radius_au: 0.0,
        period_years: 1.0, // unused for r=0
        angle: 0.0,
        color: Color::srgb(1.2, 1.0, 0.3),
        size: 3.0,
    });

    bodies.push(OrbitBody {
        name: "Mercury".into(),
        radius_au: 0.8,
        period_years: 0.24,
        angle: 0.0,
        color: Color::srgb(0.8, 0.8, 0.8),
        size: 0.9,
    });

    bodies.push(OrbitBody {
        name: "Venus".into(),
        radius_au: 1.0,
        period_years: 0.62,
        angle: 1.4,
        color: Color::srgb(1.0, 0.9, 0.4),
        size: 1.1,
    });

    bodies.push(OrbitBody {
        name: "Earth".into(),
        radius_au: 1.4,
        period_years: 1.0,
        angle: 3.0,
        color: Color::srgb(0.3, 0.6, 1.2),
        size: 1.2,
    });

    bodies.push(OrbitBody {
        name: "Mars".into(),
        radius_au: 2.0,
        period_years: 1.88,
        angle: 0.6,
        color: Color::srgb(1.1, 0.4, 0.35),
        size: 1.0,
    });

    bodies.push(OrbitBody {
        name: "Jupiter".into(),
        radius_au: 5.0,
        period_years: 11.86,
        angle: 2.1,
        color: Color::srgb(1.1, 0.8, 0.5),
        size: 1.8,
    });

    let num_bodies = bodies.len();

    // ---- Trails: one VecDeque per body ----
    let mut trails_vec = Vec::new();
    trails_vec.resize_with(num_bodies, VecDeque::new);

    commands.insert_resource(Orbits { bodies: bodies.clone() });
    commands.insert_resource(Trails { points: trails_vec });

    // ---- Docs tab default ----
    commands.insert_resource(DocTabState {
        current: DocTab::Overview,
    });

    // ---- Camera & light ----
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 25.0, 80.0).looking_at(Vec3::ZERO, Vec3::Y),
        FlyCam,
    ));

    commands.spawn((
        PointLight {
            intensity: 60_000.0,
            range: 600.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(0.0, 30.0, 0.0),
    ));

    // ---- Body meshes ----
    for (i, body) in bodies.iter().enumerate() {
        let mesh = Mesh::from(Sphere {
            radius: body.size,
            ..Default::default()
        });

        let mesh_handle = meshes.add(mesh);
        let material_handle = materials.add(StandardMaterial {
            base_color: body.color,
            emissive: body.color.into(),
            metallic: 0.1,
            perceptual_roughness: 0.5,
            ..default()
        });

        let p = if body.radius_au == 0.0 {
            Vec3::new(0.0, funnel_height(0.01), 0.0)
        } else {
            orbit_position_on_funnel(body.radius_au, body.angle)
        };

        let pos = Vec3::new(p.x, p.y + body.size * 0.5, p.z);

        commands.spawn((
            Mesh3d(mesh_handle),
            MeshMaterial3d(material_handle),
            Transform::from_translation(pos),
            BodyIndex(i),
        ));
    }
}

// =============== MINI N-BODY SETUP & UPDATE ===============

fn setup_schematic_system(mut commands: Commands) {
    // star
    let star = SchematicBody {
        mass: 1000.0,
        pos: Vec2::ZERO,
        vel: Vec2::ZERO,
        color: egui::Color32::from_rgb(255, 230, 120),
        radius_px: 10.0,
    };

    // planet 1
    let p1 = SchematicBody {
        mass: 1.0,
        pos: Vec2::new(2.0, 0.0),
        vel: Vec2::new(0.0, 9.0),
        color: egui::Color32::from_rgb(130, 190, 255),
        radius_px: 4.0,
    };

    // planet 2
    let p2 = SchematicBody {
        mass: 0.5,
        pos: Vec2::new(-4.0, 0.0),
        vel: Vec2::new(0.0, -6.0),
        color: egui::Color32::from_rgb(255, 150, 120),
        radius_px: 3.5,
    };

    commands.insert_resource(SchematicSystem {
        bodies: vec![star, p1, p2],
    });
}

fn update_schematic_nbody(time: Res<Time>, mut system: ResMut<SchematicSystem>) {
    let dt = time.delta_secs() * 0.5; // slow it down a bit
    if dt <= 0.0 {
        return;
    }

    let g: f32 = 1.0;
    let n = system.bodies.len();
    if n == 0 {
        return;
    }

    let mut accels = vec![Vec2::ZERO; n];

    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let dir = system.bodies[j].pos - system.bodies[i].pos;
            let dist2 = dir.length_squared() + 1e-3;
            let inv_dist = dist2.sqrt().recip();
            let inv_dist3 = inv_dist * inv_dist * inv_dist;

            let a = g * system.bodies[j].mass * inv_dist3 * dir;
            accels[i] += a;
        }
    }

    for (body, a) in system.bodies.iter_mut().zip(accels.iter()) {
        body.vel += *a * dt;
        body.pos += body.vel * dt;
    }
}

// =============== DOC UI OVERLAY (TABS + SCHEMATIC) ===============

fn show_equation_window_ui(
    mut contexts: EguiContexts,
    mut tab_state: ResMut<DocTabState>,
    system: Option<Res<SchematicSystem>>,
) {
    let egui_ctx = contexts.ctx_mut();

    egui::Window::new("Gravity Funnel Documentation")
        .default_width(420.0)
        .default_height(480.0)
        .resizable(true)
        .vscroll(true)
        .show(egui_ctx.expect("REASON"), |ui| {
            ui.add_space(6.0);

            // --- Tab bar ---
            ui.horizontal(|ui| {
                if ui
                    .selectable_label(tab_state.current == DocTab::Overview, "Overview")
                    .clicked()
                {
                    tab_state.current = DocTab::Overview;
                }
                if ui
                    .selectable_label(tab_state.current == DocTab::Equations, "Equations")
                    .clicked()
                {
                    tab_state.current = DocTab::Equations;
                }
                if ui
                    .selectable_label(tab_state.current == DocTab::RealPhysics, "Real Physics Demo")
                    .clicked()
                {
                    tab_state.current = DocTab::RealPhysics;
                }
            });

            ui.separator();
            ui.add_space(6.0);

            match tab_state.current {
                DocTab::Overview => {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        ui.heading("Overview");
                        ui.add_space(6.0);
                        ui.label(
"• Main 3D window: analytic circular orbits on a gravity funnel.
• The funnel encodes a softened gravitational potential.
• Bodies follow fixed-radius circular paths with pre-set periods.

This is a visual / educational model, not a full N-body
simulation. The 'Real Physics Demo' tab below runs a tiny
2D N-body system for comparison.",
                        );
                    });
                }
                DocTab::Equations => {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        ui.heading("1) Funnel Surface (Softened Newtonian Potential)");
                        ui.add_space(4.0);
                        ui.code(
"r_soft   = sqrt(r^2 + R_soft^2)
phi(r)   = -1 / r_soft
phi_edge = -1 / (R_funnel + R_soft)

height(r) = (phi(r) - phi_edge) * FUNNEL_DEPTH",
                        );

                        ui.add_space(8.0);
                        ui.heading("2) Mapping Orbit to World Coordinates");
                        ui.add_space(4.0);
                        ui.code(
"x = r * cos(theta) * SCALE
z = r * sin(theta) * SCALE
y = height(r)

y_body = y + body.size * 0.5",
                        );

                        ui.add_space(8.0);
                        ui.heading("3) Analytic Circular Orbit Motion");
                        ui.add_space(4.0);
                        ui.code(
"dt_years = delta_seconds * YEARS_PER_SECOND
omega    = 2 * pi / period_years
theta    = (theta + omega * dt_years) mod 2*pi",
                        );

                        ui.add_space(8.0);
                        ui.heading("4) Trails");
                        ui.add_space(4.0);
                        ui.code(
"push_back(position)
if trail.len() > TRAIL_LEN:
    pop_front()",
                        );

                        ui.add_space(8.0);
                        ui.heading("5) Funnel Grid");
                        ui.add_space(4.0);
                        ui.code(
"for x,z in grid:
    r = sqrt((x/SCALE)^2 + (z/SCALE)^2)
    y = height(r)
    draw line segments along X and Z",
                        );
                    });
                }
                DocTab::RealPhysics => {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        ui.heading("Mini N-body Gravity Simulation");
                        ui.add_space(4.0);
                        ui.label(
"This schematic is a live 2D N-body simulation
running only in this UI. All bodies attract
each other with a Newtonian 1/r^2 force.",
                        );

                        ui.add_space(10.0);

                        if let Some(system) = system.as_ref() {
                            let desired_size = egui::vec2(320.0, 320.0);
                            let (rect, _response) =
                                ui.allocate_exact_size(desired_size, egui::Sense::hover());
                            let painter = ui.painter_at(rect);

                            let center = rect.center();
                            let scale = 30.0;

                            // background
                            painter.rect_filled(
                                rect,
                                8.0,
                                egui::Color32::from_rgb(10, 10, 20),
                            );

                            // draw bodies
                            for b in &system.bodies {
                                let pos_screen = egui::pos2(
                                    center.x + b.pos.x * scale,
                                    center.y - b.pos.y * scale,
                                );
                                painter.circle_filled(pos_screen, b.radius_px, b.color);
                            }

                            // border
                            painter.rect_stroke(
                                rect,
                                8.0,
                                egui::Stroke::new(
                                    1.0,
                                    egui::Color32::from_rgb(120, 120, 180),
                                ),
                                egui::StrokeKind::Inside,
                            );
                        } else {
                            ui.label("No schematic system resource found.");
                        }

                        ui.add_space(8.0);
                        ui.heading("N-body Equations");
                        ui.code(
"for each body i:
    a_i = 0
    for each body j != i:
        r    = x_j - x_i
        dist = |r|
        a_i += G * m_j * r / dist^3

vel_i += a_i * dt
pos_i += vel_i * dt",
                        );
                    });
                }
            }
        });
}

// ================= ORBIT UPDATE =================

fn update_orbits_and_trails(
    time: Res<Time>,
    mut orbits: ResMut<Orbits>,
    mut trails: ResMut<Trails>,
) {
    let dt_years = time.delta_secs() * YEARS_PER_SECOND;

    for (i, body) in orbits.bodies.iter_mut().enumerate() {
        if body.radius_au > 0.0 {
            let omega = TAU / body.period_years;
            body.angle = (body.angle + omega * dt_years) % TAU;
        }

        let p = if body.radius_au == 0.0 {
            Vec3::new(0.0, funnel_height(0.01), 0.0)
        } else {
            orbit_position_on_funnel(body.radius_au, body.angle)
        };

        let deque = &mut trails.points[i];
        deque.push_back(p);
        if deque.len() > TRAIL_LEN {
            deque.pop_front();
        }
    }
}

// ================= SYNC TRANSFORMS =================

fn sync_body_transforms(orbits: Res<Orbits>, mut query: Query<(&BodyIndex, &mut Transform)>) {
    for (BodyIndex(i), mut transform) in &mut query {
        let body = &orbits.bodies[*i];

        let p = if body.radius_au == 0.0 {
            Vec3::new(0.0, funnel_height(0.01), 0.0)
        } else {
            orbit_position_on_funnel(body.radius_au, body.angle)
        };

        transform.translation = Vec3::new(p.x, p.y + body.size * 0.5, p.z);
    }
}

// ================= SPAWN RANDOM CARTOON BODY =================

fn spawn_random_body(
    mut commands: Commands,
    mut orbits: ResMut<Orbits>,
    mut trails: ResMut<Trails>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut rng_res: ResMut<RngRes>,
    keyboard: Res<ButtonInput<KeyCode>>,
) {
    if !keyboard.just_pressed(KeyCode::KeyN) {
        return;
    }

    let rng = &mut rng_res.0;

    let radius_au = rng.random_range(2.0..9.0);
    let period_years = rng.random_range(2.0..25.0);
    let angle = rng.random_range(0.0..TAU);
    let color = Color::srgb(rng.random::<f32>(), rng.random::<f32>(), rng.random::<f32>());
    let size = rng.random_range(0.7..1.4);

    let name = format!("R{}", orbits.bodies.len());

    let body = OrbitBody {
        name,
        radius_au,
        period_years,
        angle,
        color,
        size,
    };

    let idx = orbits.bodies.len();
    orbits.bodies.push(body.clone());
    trails.points.push(VecDeque::new());

    let mesh = Mesh::from(Sphere {
        radius: size,
        ..Default::default()
    });

    let mesh_handle = meshes.add(mesh);
    let material_handle = materials.add(StandardMaterial {
        base_color: color,
        emissive: color.into(),
        metallic: 0.0,
        perceptual_roughness: 0.9,
        ..default()
    });

    let p = orbit_position_on_funnel(body.radius_au, body.angle);
    commands.spawn((
        Mesh3d(mesh_handle),
        MeshMaterial3d(material_handle),
        Transform::from_translation(Vec3::new(p.x, p.y + size * 0.5, p.z)),
        BodyIndex(idx),
    ));
}

// ================= DRAW TRAILS =================

fn draw_trails(orbits: Res<Orbits>, trails: Res<Trails>, mut gizmos: Gizmos) {
    for (i, trail) in trails.points.iter().enumerate() {
        if trail.len() < 2 {
            continue;
        }

        let color = orbits.bodies[i].color;
        let (front, back) = trail.as_slices();

        for pair in front.windows(2) {
            gizmos.line(pair[0], pair[1], color);
        }
        for pair in back.windows(2) {
            gizmos.line(pair[0], pair[1], color);
        }
    }
}

// ================= DRAW FUNNEL GRID =================

fn draw_funnel(mut gizmos: Gizmos) {
    let max_r_world = FUNNEL_RADIUS_AU * SCALE;
    let step_world = 0.5 * SCALE;

    // lines parallel to X
    let mut z = -max_r_world;
    while z <= max_r_world {
        let mut x = -max_r_world;
        let mut prev: Option<Vec3> = None;

        while x <= max_r_world {
            let r_au = ((x / SCALE).powi(2) + (z / SCALE).powi(2)).sqrt();
            let y = funnel_height(r_au);
            let p = Vec3::new(x, y, z);

            if let Some(prev_p) = prev {
                gizmos.line(prev_p, p, Color::srgb(0.2, 0.4, 0.9));
            }
            prev = Some(p);

            x += step_world;
        }

        z += step_world;
    }

    // lines parallel to Z
    let mut x = -max_r_world;
    while x <= max_r_world {
        let mut z = -max_r_world;
        let mut prev: Option<Vec3> = None;

        while z <= max_r_world {
            let r_au = ((x / SCALE).powi(2) + (z / SCALE).powi(2)).sqrt();
            let y = funnel_height(r_au);
            let p = Vec3::new(x, y, z);

            if let Some(prev_p) = prev {
                gizmos.line(prev_p, p, Color::srgb(0.15, 0.3, 0.8));
            }
            prev = Some(p);

            z += step_world;
        }

        x += step_world;
    }
}

// ================= MAIN =================

fn main() {
    App::new()
        .insert_resource(RngRes(StdRng::from_os_rng()))
        .insert_resource(MovementSettings {
            sensitivity: 0.00015,
            speed: 25.0,
        })
        .add_plugins(DefaultPlugins)
        .add_plugins(NoCameraPlayerPlugin)
        .add_plugins(EguiPlugin::default())
        .add_systems(Startup, (setup, setup_schematic_system))
        .add_systems(
            Update,
            (
                update_orbits_and_trails,
                sync_body_transforms,
                spawn_random_body,
                draw_funnel,
                draw_trails,
                update_schematic_nbody,
            ),
        )
        .add_systems(EguiPrimaryContextPass, show_equation_window_ui)
        .run();
}
