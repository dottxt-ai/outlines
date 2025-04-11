This recipe demonstrates how to use the `outlines` library to extract structured event details from a text message.
We will extract the title, location, and start date and time from messages like the following:

```plaintext
Hello Kitty, my grandmother will be here, I think it's better to postpone
our appointment to review math lessons to next Monday at 2pm at the same
place, 3 avenue des tanneurs, one hour will be enough see you 😘
```

Let see how to extract the event details from the message with the MLX
library dedicated to Apple Silicon processor (M series).

```python
--8<-- "docs/cookbook/extract_event_details.py"
```

The output will be:

```plaintext
Today: Saturday 16 November 2024 and it's 10:55
```

and the extracted event information will be:

```json
{
  "title":"Math Review",
  "location":"3 avenue des tanneurs",
  "start":"2024-11-22T14:00:00Z"
}
```


To find out more about this use case, we recommend the project developped by [Joseph Rudoler](https://x.com/JRudoler) the [ICS Generator](https://github.com/jrudoler/ics-generator)
