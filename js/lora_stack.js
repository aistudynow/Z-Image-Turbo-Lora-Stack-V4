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

                this.updateVisibility = function () {
                    const lcount = this.widgets ? this.widgets.find((w) => w.name === "lora_count") : null;
                    const currentCount = lcount ? lcount.value : 1;
                    toggleWidget(this, lcount, false);

                    for (let i = 1; i <= 10; ++i) {
                        const visible = i <= currentCount;
                        const w1 = this.widgets?.find((w) => w.name === `enabled_${i}`);
                        const w2 = this.widgets?.find((w) => w.name === `lora_name_${i}`);
                        const w3 = this.widgets?.find((w) => w.name === `strength_${i}`);
                        toggleWidget(this, w1, visible);
                        toggleWidget(this, w2, visible);
                        toggleWidget(this, w3, visible);
                    }
                    if (this.setSize && this.computeSize) {
                        this.setSize([this.size[0], this.computeSize([this.size[0], this.size[1]])[1]]);
                    }
                };

                this.addWidget("button", "+ Add LoRA", null, () => {
                    const lcount = this.widgets?.find((w) => w.name === "lora_count");
                    if (lcount && lcount.value < 10) {
                        lcount.value++;
                        this.updateVisibility();
                        app.graph.setDirtyCanvas(true, true);
                    }
                });

                this.addWidget("button", "- Remove LoRA", null, () => {
                    const lcount = this.widgets?.find((w) => w.name === "lora_count");
                    if (lcount && lcount.value > 1) {
                        lcount.value--;
                        this.updateVisibility();
                        app.graph.setDirtyCanvas(true, true);
                    }
                });

                // When adding from search, execute it slightly later after the node is fully added 
                // and LiteGraph has populated the widgets list.
                setTimeout(() => {
                    if (this.updateVisibility) {
                        this.updateVisibility();
                    }
                }, 50);
            };

            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                if (onConfigure) {
                    onConfigure.apply(this, arguments);
                }
                // When loading a saved workflow, the widget values are restored here.
                if (this.updateVisibility) {
                    this.updateVisibility();
                }
            };

            // Handle when users change a connected int widget or link values
            const onConnectionsChange = nodeType.prototype.onConnectionsChange;
            nodeType.prototype.onConnectionsChange = function () {
                if (onConnectionsChange) {
                    onConnectionsChange.apply(this, arguments);
                }
                if (this.updateVisibility) {
                    this.updateVisibility();
                }
            };

            // Hook when properties/values explicitly change (like manual sliding)
            const onWidgetChanged = nodeType.prototype.onWidgetChanged;
            nodeType.prototype.onWidgetChanged = function (name, value, old_value, widget) {
                if (onWidgetChanged) {
                    onWidgetChanged.apply(this, arguments);
                }
                if (name === "lora_count" && this.updateVisibility) {
                    this.updateVisibility();
                }
            }
        }
    }
});
