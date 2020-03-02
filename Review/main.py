from Review.app import create_app
app = create_app()


def run():
    app.run(debug=True)
