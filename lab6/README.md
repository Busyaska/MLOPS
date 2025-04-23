<h3>Сборка образа:</h3>
<code>docker build -t fastapi .</code>
<hr>
<h3>Сборка контейнера</h3>
<code>docker container run --name fastapi_container -p 8080:8080 fastapi</code>
<hr>
<h3>Повторный запуск контейнера</h3>
<code>docker container start fastapi_container</code>
<hr>
<h3>Остановка контейнера</h3>
<code>docker container stop fastapi_container</code>