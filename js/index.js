window.addEventListener("load", (e) => { 
    const logo = document.getElementById("loading")
    const content = document.getElementById("content")
    content.style.display = 'none'
    setTimeout(() => {
        logo.style.display = 'none'
        content.style.display = 'block'
    }, 1000)
})