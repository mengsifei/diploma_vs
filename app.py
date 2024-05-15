from app import create_app

app = create_app()

if __name__ == '__main__':
    port = 8080
    app.run(debug=True,host='0.0.0.0',port=port)