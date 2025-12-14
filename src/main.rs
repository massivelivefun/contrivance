use contrivance::application::App;
use iced::executor;

fn main() -> iced::Result {
    iced::application(App::default, App::update, App::view)
        .title(App::title)
        .theme(App::theme)
        .subscription(App::subscription)
        .executor::<executor::Default>()
        .run()
}
