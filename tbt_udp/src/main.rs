
mod client;
mod server;

fn main() -> std::io::Result<()>{
	//toggle one of these
	//server::run()
	client::run()
}
