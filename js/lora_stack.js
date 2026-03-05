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

function getWidgetValues(widget) {
    if (!widget || !widget.options) return [];
    const values = widget.options.values;
    if (Array.isArray(values)) return values;
    if (typeof values === "function") {
        try {
            const resolved = values();
            return Array.isArray(resolved) ? resolved : [];
        } catch {
            return [];
        }
    }
    return [];
}

function normalizeLoraWidgetValue(widget) {
    if (!widget || typeof widget.value !== "string") return;

    const values = getWidgetValues(widget);
    if (!values.length) return;

    const current = widget.value;
    if (values.includes(current)) return;

    const slashNormalized = current.replace(/\\/g, "/");
    if (values.includes(slashNormalized)) {
        widget.value = slashNormalized;
        return;
    }

    const backslashNormalized = current.replace(/\//g, "\\");
    if (values.includes(backslashNormalized)) {
        widget.value = backslashNormalized;
        return;
    }

    const filename = current.split(/[\\/]/).pop();
    if (filename) {
        const suffixSlash = `/${filename}`;
        const suffixBackslash = `\\${filename}`;
        const matches = values.filter(
            (v) => typeof v === "string" && (v === filename || v.endsWith(suffixSlash) || v.endsWith(suffixBackslash))
        );
        if (matches.length === 1) {
            widget.value = matches[0];
            return;
        }
    }

    widget.value = values[0];
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
                        normalizeLoraWidgetValue(w2);
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
