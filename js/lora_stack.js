import { app } from "../../scripts/app.js";

function toggleWidget(node, widget, show) {
    if (!widget) return;
    if (show) {
        if (widget.origType !== undefined) {
            widget.type = widget.origType;
            delete widget.origType;
            if (widget.origComputeSize !== undefined) {
                widget.computeSize = widget.origComputeSize;
                delete widget.origComputeSize;
            } else {
                delete widget.computeSize;
            }
        }
    } else {
        if (widget.type !== "hidden") {
            widget.origType = widget.type;
            if (widget.hasOwnProperty("computeSize")) {
                widget.origComputeSize = widget.computeSize;
            }
            widget.type = "hidden";
            widget.computeSize = () => [0, -4];
        }
    }
}

app.registerExtension({
    name: "ZImage.TurboLoraStackV4",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "ZImageTurboLoraStackV4") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            
            nodeType.prototype.onNodeCreated = function () {
                if (onNodeCreated) {
                    onNodeCreated.apply(this, arguments);
                }
                
                const loraCountWidget = this.widgets.find((w) => w.name === "lora_count");
                
                this.updateVisibility = function() {
                    const currentCount = loraCountWidget ? loraCountWidget.value : 1;
                    toggleWidget(this, loraCountWidget, false);
                    
                    for (let i = 1; i <= 10; ++i) {
                        const visible = i <= currentCount;
                        const w1 = this.widgets.find((w) => w.name === `enabled_${i}`);
                        const w2 = this.widgets.find((w) => w.name === `lora_name_${i}`);
                        const w3 = this.widgets.find((w) => w.name === `strength_${i}`);
                        toggleWidget(this, w1, visible);
                        toggleWidget(this, w2, visible);
                        toggleWidget(this, w3, visible);
                    }
                    if (this.setSize && this.computeSize) {
                        this.setSize([this.size[0], this.computeSize([this.size[0], this.size[1]])[1]]);
                    }
                };
                
                this.addWidget("button", "+ Add LoRA", null, () => {
                    if (loraCountWidget && loraCountWidget.value < 10) {
                        loraCountWidget.value++;
                        this.updateVisibility();
                        app.graph.setDirtyCanvas(true, true);
                    }
                });
                
                this.addWidget("button", "- Remove LoRA", null, () => {
                    if (loraCountWidget && loraCountWidget.value > 1) {
                        loraCountWidget.value--;
                        this.updateVisibility();
                        app.graph.setDirtyCanvas(true, true);
                    }
                });

                requestAnimationFrame(() => {
                    if (this.updateVisibility) {
                        this.updateVisibility();
                    }
                });
            };

            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                if (onConfigure) {
                    onConfigure.apply(this, arguments);
                }
                if (this.updateVisibility) {
                    this.updateVisibility();
                }
            };
        }
    }
});
