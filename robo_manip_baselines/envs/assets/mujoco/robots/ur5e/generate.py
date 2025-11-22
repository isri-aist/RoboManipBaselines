#! /usr/bin/env python


def generate(template_file, prefix_list):
    with open(template_file, "r", encoding="utf-8") as f:
        template_text = f.read()

    for prefix in prefix_list:
        if prefix == "":
            output_file = template_file.replace(".in", "")
        else:
            output_file = template_file.replace(".in", "_" + prefix.rstrip("/"))

        output_text = template_text.format(PREFIX=prefix)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(output_text)

        print(f"Generated: {output_file}")


prefix_list = ["", "left/", "right/"]
generate("ur5e_integrated_specific_config.in.xml", prefix_list)
generate("ur5e_integrated_body.in.xml", prefix_list)
