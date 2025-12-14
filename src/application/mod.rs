pub mod cube_pipeline;
pub mod cube_primitive;
pub mod cube_program;

pub mod skybox_pipeline;
pub mod skybox_primitive;
pub mod skybox_program;

use iced::widget::{
    button, column, container, scrollable, shader, stack, text, text_input, Id,
};
use iced::{
    alignment, color, gradient, window, Background, Color,
    Element, Length, Subscription, Task, Theme, Radians
};
use crate::application::skybox_program::SkyboxProgram;
use crate::model::{Model, init_model};
use std::f32::consts::PI;
use std::time::Instant;
use std::sync::OnceLock;
use iced::advanced::widget::operation::scrollable::snap_to;

const CHAT_HEIGHT_PCT: f32 = 0.3;
const TYPING_BOX_HEIGHT: f32 = 60.0;
const NORTH_RADIANS: Radians = Radians(PI);

static SCROLLABLE_ID: OnceLock<Id> = OnceLock::new();

fn chat_scrollable_id() -> Id {
    SCROLLABLE_ID
        .get_or_init(Id::unique)
        .clone()
}

// Should I be using this struct or should I be using the CubePrimitive struct?
// The Global Uniform data (the transformation matrix)
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    // We use a 4x4 matrix for 3D transformations
    transform: [[f32; 4]; 4],
}

// Data for the vertex shader (position and color per vertex)
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    _pos: [f32; 3],
    _color: [f32; 3],
}

// TO-DO: Need to enable antialiasing for the application
pub struct App {
    messages: Vec<String>,
    input_value: String,
    model: Model,
    start_time: Instant,
}

impl Default for App {
    fn default() -> Self {
        Self {
            messages: vec![
                "System: Welcome to the chat!".into(),
                "System: Type a message below and press Enter.".into(),
            ],
            input_value: String::new(),
            model: init_model().unwrap(),
            start_time: Instant::now(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Message {
    InputChanged(String),
    SendMessage,
    Tick(Instant),
}

impl App {
    pub fn title(&self) -> String {
        String::from("Contrivance")
    }

    pub fn theme(&self) -> Theme {
        Theme::Dark
    }

    pub fn subscription(&self) -> Subscription<Message> {
        // Necessary to keep the animation loop running smoothly
        window::frames().map(Message::Tick)
    } 

    // Do I have to type state as an App? feels so weird
    pub fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::InputChanged(value) => {
                self.input_value = value;
                Task::none()
            },
            Message::SendMessage => {
                if !self.input_value.is_empty() {
                    self.messages.push(format!("You: {}", self.input_value));
                    let results = self.model.compute_input_text(&self.input_value).unwrap();
                    for result in results {
                        self.messages.push(result);
                    }
                    // self.messages.push(format!("System: {:?}", self.model.compute_input_text(&self.input_value).unwrap()));
                    self.input_value.clear();
                }
                // I need to fix the scrolling direction
                // snap_to(chat_scrollable_id(), scrollable::RelativeOffset::END.into()).into()
                Task::none()
            },
            Message::Tick(_instant) => {
                Task::none()
            },
        }
    }

    pub fn view(&self) -> Element<'_, Message> {
        // The 3D viewport
        // We pass the current time to the shader struct to animate the cube
        let viewport_3d = shader::<Message, SkyboxProgram>(SkyboxProgram {
            time: self.start_time.elapsed().as_secs_f32(),
        })
        .width(Length::Fill)
        .height(Length::Fill);

        // The UI elements
        let chat_ui = container(
            column![
                scrollable(
                    column(
                        self.messages.iter().map(|msg| {
                            // TO-DO: Add a style to the text
                            text(msg).size(18).color(color!(0x00ff00)).font(iced::Font::MONOSPACE).into()
                        })
                    )
                    .spacing(10)
                )
                .id(chat_scrollable_id())
                .height(Length::Fill)
                .width(Length::Fill),

                container(
                    iced::widget::row![
                        text_input("Type a message...", &self.input_value)
                            .on_input(Message::InputChanged)
                            .on_submit(Message::SendMessage)
                            .padding(10),
                        button("Send")
                            .on_press(Message::SendMessage)
                            .padding(10),
                    ]
                    .spacing(10)
                )
                .height(Length::Fixed(TYPING_BOX_HEIGHT))
                .style(|_theme| {
                    container::Style {
                        background: Some(Background::Color(Color::BLACK)),
                        ..container::Style::default()
                    }
                })
            ]
        )
        .width(Length::Fill)
        .padding(10)
        // Calculate the height based on the percentage configuration variable
        .height(Length::FillPortion((CHAT_HEIGHT_PCT * 100.0) as u16))
        .align_y(alignment::Vertical::Bottom)
        .style(|_theme| {
            container::Style {
                background: Some(Background::Gradient(gradient::Linear::new(NORTH_RADIANS)
                    .add_stop(0.0, color!(0, 0, 0, 0.0))
                    .add_stop(1.0, color!(0, 0, 0, 1.0))
                    .into()
                )),
                ..container::Style::default()
            }
        });

        let ui_layout = column![
            // Spacer to push the chat UI to the bottom
            container(text("")).height(Length::FillPortion(((1.0 - CHAT_HEIGHT_PCT) * 100.0) as u16)),
            chat_ui,
        ];

        stack![
            viewport_3d,
            ui_layout,
        ]
        .into()
    }
}

// Hardcoded vertices for a cube (position x,y,z, color r,g,b)
#[rustfmt::skip]
static CUBE_VERTICES: &[Vertex] = &[
    // Front face (z = 1.0, Red)
    Vertex { _pos: [-1.0, -1.0, 1.0], _color: [1.0, 0.2, 0.2] },
    Vertex { _pos: [ 1.0, -1.0, 1.0], _color: [1.0, 0.2, 0.2] },
    Vertex { _pos: [ 1.0,  1.0, 1.0], _color: [1.0, 0.2, 0.2] },
    Vertex { _pos: [-1.0, -1.0, 1.0], _color: [1.0, 0.2, 0.2] },
    Vertex { _pos: [ 1.0,  1.0, 1.0], _color: [1.0, 0.2, 0.2] },
    Vertex { _pos: [-1.0,  1.0, 1.0], _color: [1.0, 0.2, 0.2] },
    // Back face (z = -1.0, Blue)
    Vertex { _pos: [-1.0, -1.0, -1.0], _color: [0.2, 0.2, 1.0] },
    Vertex { _pos: [-1.0,  1.0, -1.0], _color: [0.2, 0.2, 1.0] },
    Vertex { _pos: [ 1.0,  1.0, -1.0], _color: [0.2, 0.2, 1.0] },
    Vertex { _pos: [-1.0, -1.0, -1.0], _color: [0.2, 0.2, 1.0] },
    Vertex { _pos: [ 1.0,  1.0, -1.0], _color: [0.2, 0.2, 1.0] },
    Vertex { _pos: [ 1.0, -1.0, -1.0], _color: [0.2, 0.2, 1.0] },
    // Top face (y = 1.0, Green)
    Vertex { _pos: [-1.0, 1.0, -1.0], _color: [0.2, 1.0, 0.2] },
    Vertex { _pos: [-1.0, 1.0,  1.0], _color: [0.2, 1.0, 0.2] },
    Vertex { _pos: [ 1.0, 1.0,  1.0], _color: [0.2, 1.0, 0.2] },
    Vertex { _pos: [-1.0, 1.0, -1.0], _color: [0.2, 1.0, 0.2] },
    Vertex { _pos: [ 1.0, 1.0,  1.0], _color: [0.2, 1.0, 0.2] },
    Vertex { _pos: [ 1.0, 1.0, -1.0], _color: [0.2, 1.0, 0.2] },
    // Bottom face (y = -1.0, Yellow)
    Vertex { _pos: [-1.0, -1.0, -1.0], _color: [1.0, 1.0, 0.2] },
    Vertex { _pos: [ 1.0, -1.0, -1.0], _color: [1.0, 1.0, 0.2] },
    Vertex { _pos: [ 1.0, -1.0,  1.0], _color: [1.0, 1.0, 0.2] },
    Vertex { _pos: [-1.0, -1.0, -1.0], _color: [1.0, 1.0, 0.2] },
    Vertex { _pos: [ 1.0, -1.0,  1.0], _color: [1.0, 1.0, 0.2] },
    Vertex { _pos: [-1.0, -1.0,  1.0], _color: [1.0, 1.0, 0.2] },
    // Right face (x = 1.0, Cyan)
    Vertex { _pos: [1.0, -1.0, -1.0], _color: [0.2, 1.0, 1.0] },
    Vertex { _pos: [1.0,  1.0, -1.0], _color: [0.2, 1.0, 1.0] },
    Vertex { _pos: [1.0,  1.0,  1.0], _color: [0.2, 1.0, 1.0] },
    Vertex { _pos: [1.0, -1.0, -1.0], _color: [0.2, 1.0, 1.0] },
    Vertex { _pos: [1.0,  1.0,  1.0], _color: [0.2, 1.0, 1.0] },
    Vertex { _pos: [1.0, -1.0,  1.0], _color: [0.2, 1.0, 1.0] },
    // Left face (x = -1.0, Magenta)
    Vertex { _pos: [-1.0, -1.0, -1.0], _color: [1.0, 0.2, 1.0] },
    Vertex { _pos: [-1.0, -1.0,  1.0], _color: [1.0, 0.2, 1.0] },
    Vertex { _pos: [-1.0,  1.0,  1.0], _color: [1.0, 0.2, 1.0] },
    Vertex { _pos: [-1.0, -1.0, -1.0], _color: [1.0, 0.2, 1.0] },
    Vertex { _pos: [-1.0,  1.0,  1.0], _color: [1.0, 0.2, 1.0] },
    Vertex { _pos: [-1.0,  1.0, -1.0], _color: [1.0, 0.2, 1.0] },
];
