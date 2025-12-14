use iced::mouse::{self, Interaction};
use iced::widget::shader;
use iced::widget::Action;
use crate::application::cube_primitive::CubePrimitive;

// The Cube State Data
// How can I use CubeProgramState
#[derive(Debug, Default)]
pub struct CubeProgram {
    pub time: f32,
}

#[derive(Debug, Default)]
pub struct CubeState {}

impl<Message> shader::Program<Message> for CubeProgram {
    // The internal state (State and Primitive) that the program keeps
    type State = CubeState;
    type Primitive = CubePrimitive;

    fn draw(
        &self,
        _state: &Self::State,
        _cursor: iced::mouse::Cursor,
        bounds: iced::Rectangle,
        // impl vs Box<dyn T> for trait object, what is the difference?
    ) -> Self::Primitive {
        // println!("Drawing cube: {:?}", bounds);
        let aspect = bounds.width / bounds.height;
        let projection = glam::Mat4::perspective_rh(45.0f32.to_radians(), aspect, 0.1, 100.0);
        let view = glam::Mat4::look_at_rh(
            glam::Vec3::new(0.0, 2.0, 5.0),
            glam::Vec3::ZERO,
            glam::Vec3::Y,
        );

        // Rotate based on time
        let model = glam::Mat4::from_rotation_y(self.time * 0.5)
                    * glam::Mat4::from_rotation_x(self.time * 0.3);

        let transform_matrix = projection * view * model;

        CubePrimitive {
            transform: transform_matrix.to_cols_array_2d(),
        }
    }

    // Update the internal Self::State of the Program
    fn update(
        &self,
        _state: &mut Self::State,
        _event: &iced::Event,
        _bounds: iced::Rectangle,
        _cursor: iced::mouse::Cursor,
    ) -> Option<Action<Message>> {
        None
    }

    fn mouse_interaction(
        &self,
        _state: &Self::State,
        _bounds: iced::Rectangle,
        _cursor: mouse::Cursor,
    ) -> Interaction {
        Interaction::default()
    }
}
