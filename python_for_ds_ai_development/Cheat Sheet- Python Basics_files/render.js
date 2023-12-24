function httpGet(theUrl) {
    var xmlHttp = new XMLHttpRequest();
    xmlHttp.open("GET", theUrl, false); // false for synchronous request
    xmlHttp.send(null);
    return xmlHttp.responseText;
}

function addAudioWidget(doc, audio) {
    let outdatedText = "The audio track is being regenerated at the moment, so the currently loaded audio may be outdated. We recommend restarting the page in a couple of minutes to access the latest audio content. Please, contact support if this warning persists for too long."
    let invalidText = "We've encountered an issue with the audio file. We suggest restarting the page to see if it resolves the problem. If you continue to see this warning, please reach out to our support for further assistance."
    if (audio["url"]) {
        // adding audio section at the top of the preview container
        doc.previewContainer.before(`
                <div id="audio-container">
                    <div class="collapse ${audio["isValid"] ? "show" : ""}" id="audio-widget-container">
                        <div id="audio-widget" class="${audio["isValid"] ? "" : "disabled"}">
                            <i id="audioWarning" class="fa fa-lg fa-exclamation-circle d-none ${!audio["isValid"] ? "invalid" : "outdated" }" aria-hidden="true" data-toggle="popover" data-trigger="hover" data-placement="top" title="Warning" data-content="${!audio["isValid"] ? invalidText : outdatedText}"></i>
                            <audio crossorigin playsinline>
                                <source src="${audio["url"]}?${Date.now()}" type="audio/mp3">
                            </audio>
                        </div>
                    </div>
                </div>
        `)

        // Add toolbar button for audio section toggle
        let tools = doc.preview.find('.toolbar div.tools');
        if (doc.settings.tool_type == "instructional-lab") {
            tools.children().first().after(
                `<button id="audioToggleButton" class="tool-icon ${audio["isValid"] ? "" : "collapsed"}" type="button" title="Toggle Audio Section" data-toggle="collapse" aria-expanded="${audio["isValid"] ? "true" : "false"}" aria-controls="#audio-widget-container" data-target="#audio-widget-container">
                    <i class="fa fa-bullhorn pr-2"></i>
                </button>`
            )
        }

        // audio widget setup
        window.addEventListener("load", (event) => {
            const player = new Plyr('audio', {});
            window.player = player;
            window.player.on("ready", () => {
                $('#audioWarning').popover({
                    container: "#audio-container"
                });
                if (audio["isOutdated"] || !audio["isValid"]) {
                    document.getElementById("audioWarning").classList.remove("d-none")
                }
            })
        });
    }
}