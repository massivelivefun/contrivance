use iced::mouse::{self, Interaction};
use iced::widget::shader;
use iced::widget::Action;
use crate::application::skybox_primitive::SkyboxPrimitive;

// The Skybox State Data
// How can I use SkyboxProgramState
#[derive(Debug, Default)]
pub struct SkyboxProgram {
    pub time: f32,
}

#[derive(Debug, Default)]
pub struct SkyboxState {}

impl<Message> shader::Program<Message> for SkyboxProgram {
    // The internal state (State and Primitive) that the program keeps
    type State = SkyboxState;
    type Primitive = SkyboxPrimitive;

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

        let scale = glam::Mat4::from_scale(glam::Vec3::splat(0.5));
        // Rotate based on time
        let model = glam::Mat4::from_rotation_y(self.time * 0.5)
                    * glam::Mat4::from_rotation_x(self.time * 0.3);
        let transform_matrix = projection * view * scale * model;

        let skybox_model = glam::Mat4::from_rotation_y(self.time * 0.05)
                    * glam::Mat4::from_rotation_x(self.time * 0.025);
        let view_rotation_only = glam::Mat4::from_mat3(glam::Mat3::from_mat4(view));
        let view_rot_only_proj = projection * view_rotation_only * skybox_model;

        SkyboxPrimitive {
            cube_transform: transform_matrix.to_cols_array_2d(),
            skybox_transform: view_rot_only_proj.to_cols_array_2d(),
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
