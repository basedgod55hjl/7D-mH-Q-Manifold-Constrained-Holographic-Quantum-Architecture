// File: projects/sovereign_assistant/src/main.rs
// Sovereign Terminal Interface (STI) - 7D-MHQL Cockpit
// Unified 7D Hardware Matrix & Autonomous Tool Loop

use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::{Backend, CrosstermBackend},
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, Paragraph, Tabs, Wrap},
    Frame, Terminal,
};
use std::{
    io,
    sync::mpsc,
    thread,
    time::{Duration, Instant},
};

mod hardware;
mod tokenizer;
mod tools;

use hardware::{HardwareMapper, HardwareStats};

struct App {
    input: String,
    history: Vec<(String, String)>,
    logs: Vec<String>,
    current_tab: usize,
    hardware_stats: HardwareStats,
    auto_execute: bool,
    thinking: bool,
}

impl App {
    fn new() -> Self {
        Self {
            input: String::new(),
            history: vec![(
                "assistant".to_string(),
                "Sovereign 7D Cockpit: ONLINE.\nFreedom Mode: ENABLED.".to_string(),
            )],
            logs: vec!["System initialized.".to_string()],
            current_tab: 0,
            hardware_stats: HardwareStats {
                cpu_usage: 0.0,
                core_usages: Vec::new(),
                gpus: Vec::new(),
                disks: Vec::new(),
                mem_used: 0,
                mem_total: 1,
                manifold_stability: 0.01,
                phi_delta: 0.0,
            },
            auto_execute: true,
            thinking: false,
        }
    }
}

#[tokio::main]
async fn main() -> io::Result<()> {
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Create app state
    let app = App::new();
    let (tx, rx) = mpsc::channel();

    // Spawn hardware monitoring thread (high priority)
    thread::spawn(move || {
        let mut mapper = HardwareMapper::new();
        loop {
            let stats = mapper.refresh();
            if tx.send(stats).is_err() {
                break;
            }
            thread::sleep(Duration::from_millis(500));
        }
    });

    // Run terminal UI loop
    let res = run_app(&mut terminal, app, rx).await;

    // Cleanup
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    if let Err(err) = res {
        println!("{:?}", err);
    }

    Ok(())
}

async fn run_app<B: Backend>(
    terminal: &mut Terminal<B>,
    mut app: App,
    rx: mpsc::Receiver<HardwareStats>,
) -> io::Result<()> {
    let tick_rate = Duration::from_millis(100);
    let mut last_tick = Instant::now();

    loop {
        terminal.draw(|f| ui(f, &mut app))?;

        let timeout = tick_rate
            .checked_sub(last_tick.elapsed())
            .unwrap_or_else(|| Duration::from_secs(0));

        if event::poll(timeout)? {
            if let Event::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        return Ok(());
                    }
                    KeyCode::Tab => {
                        app.current_tab = (app.current_tab + 1) % 2;
                    }
                    KeyCode::Enter => {
                        if !app.input.is_empty() {
                            let input = app.input.clone();
                            app.history.push(("user".to_string(), input.clone()));
                            app.logs.push(format!("USER: {}", input));
                            app.input.clear();
                            app.logs.push(format!("Analyzing input for 7D tools..."));
                        }
                    }
                    KeyCode::Char(c) => {
                        app.input.push(c);
                    }
                    KeyCode::Backspace => {
                        app.input.pop();
                    }
                    _ => {}
                }
            }
        }

        while let Ok(stats) = rx.try_recv() {
            app.hardware_stats = stats;
        }

        if last_tick.elapsed() >= tick_rate {
            last_tick = Instant::now();
        }
    }
}

fn ui(f: &mut Frame, app: &mut App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Header
            Constraint::Length(3), // Tabs
            Constraint::Min(10),   // Content
            Constraint::Length(3), // Input
        ])
        .split(f.size());

    // 1. Header
    let header_spans = vec![
        Span::styled(
            " 🔮 SOVEREIGN",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            " 7D-MHQL COCKPIT",
            Style::default()
                .fg(Color::Magenta)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" | MODE: ", Style::default().fg(Color::White)),
        if app.auto_execute {
            Span::styled(
                "FREEDOM (AUTO)",
                Style::default()
                    .fg(Color::LightGreen)
                    .add_modifier(Modifier::BOLD),
            )
        } else {
            Span::styled("MANUAL", Style::default().fg(Color::Yellow))
        },
        Span::styled(" | ", Style::default().fg(Color::DarkGray)),
        if app.thinking {
            Span::styled(
                "🧠 THINKING...",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::ITALIC),
            )
        } else {
            Span::styled("💎 READY", Style::default().fg(Color::Green))
        },
        Span::styled(" | STABILITY: ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            format!("{:.4}", app.hardware_stats.manifold_stability),
            if app.hardware_stats.manifold_stability < 0.005 {
                Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::Cyan)
            },
        ),
        Span::styled(" | Φ-Δ: ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            format!("{:.6}", app.hardware_stats.phi_delta),
            Style::default().fg(Color::Yellow),
        ),
    ];

    let header = Paragraph::new(Line::from(header_spans)).block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray)),
    );
    f.render_widget(header, chunks[0]);

    // 2. Tabs
    let titles = vec!["  SYSTRAY  ", "  MANIFOLD LOGS  "];
    let tabs = Tabs::new(titles)
        .block(Block::default().borders(Borders::ALL).title(" Nav "))
        .select(app.current_tab)
        .style(Style::default().fg(Color::DarkGray))
        .highlight_style(
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        );
    f.render_widget(tabs, chunks[1]);

    // 3. Main View
    if app.current_tab == 0 {
        render_systray(f, app, chunks[2]);
    } else {
        render_logs(f, app, chunks[2]);
    }

    // 4. Input Field
    let input = Paragraph::new(app.input.as_str()).block(
        Block::default()
            .borders(Borders::ALL)
            .title(Span::styled(
                " Command Channel (⌘) ",
                Style::default().fg(Color::Magenta),
            ))
            .border_style(Style::default().fg(Color::Magenta)),
    );
    f.render_widget(input, chunks[3]);
}

fn render_systray(f: &mut Frame, app: &App, area: Rect) {
    let main_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Min(30), Constraint::Length(50)])
        .split(area);

    let mut history_lines = Vec::new();
    for (role, content) in &app.history {
        let (role_styled, content_color) = match role.as_str() {
            "user" => (
                Span::styled(
                    " USER ",
                    Style::default()
                        .bg(Color::Green)
                        .fg(Color::Black)
                        .add_modifier(Modifier::BOLD),
                ),
                Color::White,
            ),
            "assistant" => (
                Span::styled(
                    " 7D-CORE ",
                    Style::default()
                        .bg(Color::Blue)
                        .fg(Color::White)
                        .add_modifier(Modifier::BOLD),
                ),
                Color::Cyan,
            ),
            _ => (
                Span::styled(
                    " DBG ",
                    Style::default().bg(Color::DarkGray).fg(Color::White),
                ),
                Color::DarkGray,
            ),
        };
        history_lines.push(Line::from(vec![
            role_styled,
            Span::raw(" "),
            Span::styled(content, Style::default().fg(content_color)),
        ]));
        history_lines.push(Line::from(""));
    }
    f.render_widget(
        Paragraph::new(history_lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" Holographic Neural Buffer "),
            )
            .wrap(Wrap { trim: false }),
        main_chunks[0],
    );

    let hw_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(5),
            Constraint::Length(4),
            Constraint::Min(10),
            Constraint::Length(3),
        ])
        .split(main_chunks[1]);

    let mut core_line = vec![
        Span::styled(
            " CORES ",
            Style::default()
                .bg(Color::Magenta)
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(" "),
    ];
    for (i, usage) in app.hardware_stats.core_usages.iter().enumerate() {
        let color = if *usage > 80.0 {
            Color::Red
        } else if *usage > 50.0 {
            Color::Yellow
        } else {
            Color::Green
        };
        let bar_char = if *usage > 75.0 {
            "█"
        } else if *usage > 50.0 {
            "▆"
        } else if *usage > 25.0 {
            "▃"
        } else {
            "▁"
        };
        core_line.push(Span::styled(bar_char, Style::default().fg(color)));
        if (i + 1) % 4 == 0 && i < 15 {
            core_line.push(Span::raw(" "));
        }
    }
    core_line.push(Span::styled(
        format!(" {:.0}%", app.hardware_stats.cpu_usage),
        Style::default().fg(Color::Cyan),
    ));

    f.render_widget(
        Paragraph::new(vec![
            Line::from(core_line),
            Line::from(vec![Span::styled(
                "   Ryzen 7 4800H: 8C/16T @ 2.9-4.2GHz",
                Style::default()
                    .fg(Color::DarkGray)
                    .add_modifier(Modifier::ITALIC),
            )]),
        ])
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" CPU Φ-Matrix "),
        ),
        hw_chunks[0],
    );

    let mem_pct =
        (app.hardware_stats.mem_used as f64 / app.hardware_stats.mem_total as f64 * 100.0) as u32;
    let mem_bar_width = (mem_pct as f32 / 100.0 * 25.0) as usize;
    let mem_bar = "█".repeat(mem_bar_width) + &" ".repeat(25_usize.saturating_sub(mem_bar_width));
    f.render_widget(
        Paragraph::new(vec![Line::from(vec![
            Span::styled(
                " RAM ",
                Style::default()
                    .bg(Color::Blue)
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(" ["),
            Span::styled(
                mem_bar,
                Style::default().fg(if mem_pct > 80 {
                    Color::Red
                } else {
                    Color::Green
                }),
            ),
            Span::raw(format!(
                "] {:.1}GB / {:.1}GB ({}%)",
                app.hardware_stats.mem_used as f64 / 1e9,
                app.hardware_stats.mem_total as f64 / 1e9,
                mem_pct
            )),
        ])])
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(" DDR4-3200 Memory "),
        ),
        hw_chunks[1],
    );

    let mut silicon_lines = Vec::new();
    for (i, gpu) in app.hardware_stats.gpus.iter().enumerate() {
        let v_color = if gpu.vendor == "AMD" {
            Color::Red
        } else {
            Color::LightGreen
        };
        silicon_lines.push(Line::from(vec![
            Span::styled(
                format!(" GPU{} ", i),
                Style::default()
                    .bg(v_color)
                    .fg(Color::Black)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(format!(" {} ", gpu.vendor), Style::default().fg(v_color)),
            Span::styled(
                &gpu.name,
                Style::default()
                    .fg(Color::Gray)
                    .add_modifier(Modifier::ITALIC),
            ),
        ]));
        let load_w = (gpu.load as f32 / 100.0 * 20.0) as usize;
        let load_b = "█".repeat(load_w) + &" ".repeat(20_usize.saturating_sub(load_w));
        silicon_lines.push(Line::from(vec![
            Span::raw("  LOAD: ["),
            Span::styled(
                load_b,
                Style::default().fg(if gpu.load > 80 {
                    Color::Red
                } else {
                    Color::Green
                }),
            ),
            Span::raw(format!("] {}%", gpu.load)),
        ]));
        silicon_lines.push(Line::from(vec![
            Span::styled("  VRAM: ", Style::default().fg(Color::White)),
            Span::styled(
                format!("{:.2}GB", gpu.mem_used as f64 / 1e9),
                Style::default().fg(Color::Blue),
            ),
            Span::raw(" / "),
            Span::styled(
                format!("{:.2}GB", gpu.mem_total as f64 / 1e9),
                Style::default().fg(Color::DarkGray),
            ),
        ]));
        silicon_lines.push(Line::from(vec![
            Span::styled("  SHRD: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{:.1}GB", gpu.shared_mem_used as f64 / 1e9),
                Style::default().fg(Color::Magenta),
            ),
            Span::raw(" / "),
            Span::styled(
                format!("{:.1}GB", gpu.shared_mem_total as f64 / 1e9),
                Style::default().fg(Color::DarkGray),
            ),
        ]));
        silicon_lines.push(Line::from(""));
    }
    for disk in &app.hardware_stats.disks {
        let d_pct = (disk.used as f64 / disk.total as f64 * 100.0) as u32;
        silicon_lines.push(Line::from(vec![
            Span::styled(
                " DSK ",
                Style::default()
                    .bg(Color::White)
                    .fg(Color::Black)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!(" {} ", disk.name),
                Style::default().fg(Color::White),
            ),
            Span::styled(
                format!(
                    "{}% ({:.1}GB free)",
                    d_pct,
                    (disk.total - disk.used) as f64 / 1e9
                ),
                Style::default().fg(Color::DarkGray),
            ),
        ]));
    }
    f.render_widget(
        Paragraph::new(silicon_lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" 7D Neural Substrate (AMD+NVIDIA+SSD) "),
            )
            .wrap(Wrap { trim: false }),
        hw_chunks[2],
    );

    f.render_widget(
        Gauge::default()
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" 💎 7D DISCOVERY CORE (STABILITY) "),
            )
            .gauge_style(Style::default().fg(Color::Cyan).bg(Color::Black))
            .ratio((app.hardware_stats.manifold_stability / 0.01).min(1.0) as f64)
            .label(format!(
                "{:.4} / 0.0100 S²",
                app.hardware_stats.manifold_stability
            )),
        hw_chunks[3],
    );
}

fn render_logs(f: &mut Frame, app: &App, area: Rect) {
    let mut log_lines = Vec::new();
    for log in app.logs.iter().rev() {
        log_lines.push(Line::from(vec![
            Span::styled(
                " [λ] ",
                Style::default()
                    .fg(Color::Magenta)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(log, Style::default().fg(Color::White)),
        ]));
    }
    f.render_widget(
        Paragraph::new(log_lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" Autonomous Execution Trace "),
            )
            .wrap(Wrap { trim: true }),
        area,
    );
}
